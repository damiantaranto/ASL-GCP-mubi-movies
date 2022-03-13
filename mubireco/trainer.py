import os
import subprocess
import sys
import time
import argparse
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tempfile
import json

from mubireco.data.train import TrainDataset
from mubireco.data.inference import InferenceDataset

from mubireco.utils.configuration import Configuration

from mubireco.model.model_definition import create_model


def train_and_evaluate(hparams):
    batch_size = hparams["batch_size"]  # 10000
    num_evals = hparams["num_evals"]  # 100
    lr = hparams["lr"]  # 0.1
    bucket_name = hparams["bucket_name"]
    timestamp = hparams["timestamp"]
    timestamp_train = hparams["timestamp_train"]
    embedding_dim = hparams["embedding_dim"]
    num_test_sample = hparams["num_test_sample"]
    output_dir = os.path.join(f"gs://{bucket_name}", timestamp, timestamp_train)

    config = Configuration("mubireco/config/config.yaml")

    ds_train = TrainDataset(config).read_tf_dataset()
    ds_inf = InferenceDataset(config).read_tf_dataset()

    num_train_sample = len(ds_train) - num_test_sample

    tf.random.set_seed(42)
    ds_train = ds_train.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    ds_train_shuffled = ds_train.take(num_train_sample)
    ds_val_shuffled = ds_train.skip(num_train_sample)

    cached_train = ds_train_shuffled.batch(batch_size).repeat().cache()
    cached_val = ds_val_shuffled.batch(batch_size).cache()

    steps_per_epoch = len(ds_train) // (batch_size * num_evals)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, "checkpoints"), save_weights_only=True, verbose=1
    )
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        os.path.join(output_dir, "tensorboard"), histogram_freq=1
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                         monitor="factorized_top_k/top_100_categorical_accuracy")

    # For MirroredStrategy #
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # End #
        model = create_model(batch_size, embedding_dim, config)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=lr))

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    cached_train = cached_train.with_options(options)
    cached_val = cached_val.with_options(options)

    start = time.time()
    history = model.fit(
        cached_train,
        # validation_data=cached_val,
        steps_per_epoch=steps_per_epoch,
        epochs=num_evals,
        verbose=1,  # 0=silent, 1=progress bar, 2=one line per epoch
        callbacks=[checkpoint_cb, tensorboard_cb]  # , early_stopping_cb],
    )
    print("Training time with single GPUs: {}".format(time.time() - start))

    results = model.evaluate(cached_val, return_dict=True)

    filename_results = "eval_results.json"
    with open(filename_results, "w") as f:
        json.dump(results, f)
    gcs_model_path = os.path.join(output_dir, filename_results)
    subprocess.check_call(["gsutil", "cp", filename_results, gcs_model_path], stderr=sys.stdout)
    print(f"Saved model in: {gcs_model_path}")

    index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)

    index.index_from_dataset(
        tf.data.Dataset.zip(
            (ds_inf.map(lambda x: x["user_id"]).batch(batch_size),
             ds_inf.batch(batch_size).map(model.candidate_model)))
    )

    # example, to keep otherwise bug in saved model
    _, _ = index(tf.constant([42]))

    with tempfile.TemporaryDirectory() as tmp:
        # Save the index.
        tf.saved_model.save(index, os.path.join(output_dir, "models"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", help="Batch size for training steps", type=int, default=10000,
    )
    parser.add_argument(
        "--lr", help="learning rate for optimizer", type=float, default=0.01
    )
    parser.add_argument(
        "--embedding_dim", help="Embedding dimension", type=int, default=32
    )
    parser.add_argument(
        "--num_evals", help="Number of times to evaluate model on eval data training.", type=int, default=50,
    )
    parser.add_argument(
        "--bucket_name", help="GCS bucket name", required=True,
    )
    parser.add_argument(
        "--timestamp", help="Data timestamp", required=True,
    )
    parser.add_argument(
        "--timestamp_train", help="Train timestamp", required=True,
    )
    parser.add_argument(
        "--num_test_sample", help="Number of test samples", type=int, default=10000
    )

    args, _ = parser.parse_known_args()

    hparams = args.__dict__
    train_and_evaluate(hparams)
