import time
import argparse
import tensorflow as tf

from mubireco.data.train import TrainDataset
from mubireco.data.validation import ValidationDataset

from mubireco.utils.configuration import Configuration

from mubireco.model.model_definition import create_model


def train_and_evaluate(hparams):
    config = Configuration("mubireco/config/config.yaml")

    ds_train = TrainDataset(config).read_tf_dataset()
    ds_val = ValidationDataset(config).read_tf_dataset()

    batch_size = hparams["batch_size"]  # 10000
    num_evals = hparams["num_evals"]  # 100
    lr = hparams["lr"]  # 0.1
    bucket_name = hparams["bucket_name"]
    timestamp = hparams["timestamp"]
    embedding_dim = hparams["embedding_dim"]

    cached_train = ds_train.shuffle(10_000, seed=42).batch(batch_size).cache()
    cached_val = ds_val.batch(batch_size).cache()

    steps_per_epoch = len(ds_train) // (batch_size * num_evals)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        f"gs://{bucket_name}/{timestamp}/checkpoints", save_weights_only=True, verbose=1
    )
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        f"gs://{bucket_name}/{timestamp}/tensorboard", histogram_freq=1
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, monitor="factorized_top_k/top_100_categorical_accuracy")

    # For MirroredStrategy #
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
    # End #
        model = create_model(batch_size, embedding_dim, config)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=lr))

    start = time.time()
    history = model.fit(
        cached_train,
        validation_data=cached_val,
        steps_per_epoch=steps_per_epoch,
        epochs=num_evals,
        verbose=1,  # 0=silent, 1=progress bar, 2=one line per epoch
        callbacks=[checkpoint_cb, tensorboard_cb, early_stopping_cb],
    )
    print("Training time with single GPUs: {}".format(time.time() - start))
    
    model.save(f"gs://{bucket_name}/{timestamp}/models")


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
        "--timestamp", help="Data timestampo", required=True,
    )

    args, _ = parser.parse_known_args()

    hparams = args.__dict__
    train_and_evaluate(hparams)
