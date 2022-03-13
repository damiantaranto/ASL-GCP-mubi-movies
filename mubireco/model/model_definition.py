import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from mubireco.data.movies import MoviesDataset
from mubireco.data.train import TrainDataset


class CandidateEncoder(tf.keras.layers.Layer):
    def __init__(self, unique_movie_ids, embedding_dimension):
        super().__init__()
        self._embedding_model = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=unique_movie_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension),
        ])

        self._norm_model_years = tf.keras.layers.LayerNormalization(axis=-1)
        self._norm_model_scores = tf.keras.layers.LayerNormalization(axis=-1)

        self._feature_columns = {
            "user_eligible_for_trial": tf.feature_column.numeric_column("user_eligible_for_trial"),
            "user_has_payment_method": tf.feature_column.numeric_column("user_has_payment_method"),
            "user_subscriber": tf.feature_column.numeric_column("user_subscriber"),
            "user_trialist": tf.feature_column.numeric_column("user_trialist"),
        }

        self._gru_encoder = tf.keras.layers.GRU(units=embedding_dimension, recurrent_initializer="glorot_uniform")

        self._dense_model = tf.keras.Sequential([
            tf.keras.layers.DenseFeatures(self._feature_columns.values()),
            tf.keras.layers.LayerNormalization(axis=1),
            tf.keras.layers.Dense(4, activation="relu")
        ])

        self._proj_layer = tf.keras.layers.Dense(embedding_dimension, activation="relu")

    def call(self, features):
        seq_embedding = self._embedding_model(features["previous_movie_ids"])
        seq_scores = tf.expand_dims(self._norm_model_scores(features["previous_score"]), -1)

        concat_sequence = tf.concat([seq_embedding, seq_scores], axis=-1)

        encoder = self._gru_encoder(concat_sequence)

        concat_encoder = tf.concat([encoder, self._dense_model(features)], axis=-1)

        proj_layer = self._proj_layer(concat_encoder)

        return proj_layer


class MubiMoviesModel(tfrs.Model):

    def __init__(self, task, candidate_model, query_model):
        super().__init__()

        self.query_model = query_model
        self.candidate_model = candidate_model

        self._task = task

    def compute_loss(self, features, training=False):
        query = features.pop("movie_id")

        query_encoder = self.query_model(query)
        candidate_encoder = self.candidate_model(features)

        return self._task(query_encoder, candidate_encoder, compute_metrics=not training)


def create_model(batch_size, embedding_dimension, config):
    df_movies = MoviesDataset(config).read_tf_dataset()
    ds_train = TrainDataset(config).read_tf_dataset()
    unique_movie_ids = np.unique(np.concatenate(list(df_movies.batch(batch_size).map(lambda x: x["movie_id"]))))
    unique_user_ids = np.unique(np.concatenate(list(ds_train.batch(batch_size).map(lambda x: x["user_id"]))))

    query_model = tf.keras.Sequential([
        tf.keras.layers.IntegerLookup(vocabulary=unique_movie_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension)
    ])

    candidate_model = CandidateEncoder(unique_movie_ids, embedding_dimension)

    metrics = tfrs.metrics.FactorizedTopK(
        candidates=ds_train.batch(batch_size).map(candidate_model),
        metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=100, name=f"factorized_top_k/top_100_categorical_accuracy")]
    )

    task = tfrs.tasks.Retrieval(
        metrics=metrics
    )

    return MubiMoviesModel(task, candidate_model, query_model)
