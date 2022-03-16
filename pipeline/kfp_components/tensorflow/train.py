from kfp.v2.dsl import Artifact, Input, Output, Dataset, Model, component
from pipeline.kfp_components.dependencies import PYTHON37, TENSORFLOW, TENSORFLOW_RECOMMENDERS, PANDAS, GOOGLE_CLOUD_BIGQUERY, PYARROW, GOOGLE_CLOUD_STORAGE

@component(base_image="gcr.io/qwiklabs-gcp-04-424f1fdacc59/moviecust:0.1", packages_to_install=[TENSORFLOW, TENSORFLOW_RECOMMENDERS, PANDAS, GOOGLE_CLOUD_BIGQUERY, PYARROW, GOOGLE_CLOUD_STORAGE])
def train_tensorflow_model(
    data_root: str,
    movies_output_filename: str,
    train_output_filename: str,
    inference_output_filename: str,
    artifact_store: str,
    timestamp: str,
    bucket_name: str,
    model: Output[Model],
):
    import os
    import json
    import datetime
    import tensorflow as tf
    import tensorflow_recommenders as tfrs
    import time
    import subprocess
    import sys
    import tempfile
    
    from google.cloud import storage

    from mubireco.model.model_definition import create_model
    from mubireco.data.train import TrainDataset
    from mubireco.data.inference import InferenceDataset

    from mubireco.utils.configuration import Configuration

    batch_size = 10000
    num_evals = 100
    lr = 0.1
    timestamp_train = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    embedding_dim = 32
    num_test_sample = 4

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

    # Instantiates a client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    gcs_model_path = os.path.join(timestamp, timestamp_train, "eval_results.json")
    blob = bucket.blob(gcs_model_path)
    blob.upload_from_string(data=json.dumps(results), content_type='application/json')
    print(f"Saved model in: {gcs_model_path}")

    model.path = os.path.join(timestamp, timestamp_train)
    model.save(model.path, save_format="tf")

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