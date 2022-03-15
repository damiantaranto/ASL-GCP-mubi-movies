from kfp.v2.dsl import Artifact, Input, Output, Dataset, Model, component
from pipeline.kfp_components.dependencies import PYTHON37, TENSORFLOW, TENSORFLOW_RECOMMENDERS


@component(base_image=PYTHON37, packages_to_install=[TENSORFLOW, TENSORFLOW_RECOMMENDERS])
def train_tensorflow_model(
    data_root: str,
    movies_output_filename: str,
    train_output_filename: str,
    inference_output_filename: str,
    artifact_store: str,
):
    import os
    import json
    import datetime
    import tensorflow as tf
    import numpy as np
    import tensorflow_recommenders as tfrs
    import time
    import subprocess
    import sys
    import tempfile

    batch_size = 10000
    num_evals = 100
    lr = 0.1
    timestamp_train = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    embedding_dim = 32
    num_test_sample = 1000
    output_dir = os.path.join(artifact_store, timestamp_train)

    movies_output_path = os.path.join(data_root, "movies", movies_output_filename)
    train_output_path = os.path.join(data_root, "train", train_output_filename)
    inference_output_path = os.path.join(data_root, "inference", inference_output_filename)

    def read_tf_dataset(output_path) -> tf.data.Dataset:
        return tf.data.experimental.load(output_path)

    def CandidateEncoder(unique_movie_ids, embedding_dimension, features):
        embedding_movie_ids = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=unique_movie_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension),
        ])

        gru_encoder = tf.keras.layers.GRU(units=embedding_dimension, recurrent_initializer="glorot_uniform")

        seq_embedding_movie_ids = embedding_movie_ids(features["previous_movie_ids"])

        encoder = gru_encoder(seq_embedding_movie_ids)

        return encoder


    def MubiMoviesModel(task, candidate_model, query_model, features, training=False):
        query = features.pop("movie_id")

        query_encoder = query_model(query)
        candidate_encoder = candidate_model(features)

        return task(query_encoder, candidate_encoder, compute_metrics=not training)



    def create_model(batch_size, embedding_dimension):
        df_movies = read_tf_dataset(movies_output_path)
        ds_train = read_tf_dataset(train_output_path)
        unique_movie_ids = np.unique(np.concatenate(list(df_movies.batch(batch_size).map(lambda x: x["movie_id"]))))

        query_model = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=unique_movie_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension)
        ])

        candidate_model = CandidateEncoder(unique_movie_ids, embedding_dimension)

        metrics = tfrs.metrics.FactorizedTopK(
            candidates=ds_train.batch(batch_size).map(candidate_model),
            metrics=[
                tf.keras.metrics.TopKCategoricalAccuracy(k=100, name=f"factorized_top_k/top_100_categorical_accuracy")]
        )

        task = tfrs.tasks.Retrieval(
            metrics=metrics
        )

        return MubiMoviesModel(task, candidate_model, query_model)

    ds_train = read_tf_dataset(train_output_path)
    ds_inf = read_tf_dataset(inference_output_path)

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
        model = create_model(batch_size, embedding_dim)
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