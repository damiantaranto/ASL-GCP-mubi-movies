import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from mubireco.data.movies import MoviesDataset
from mubireco.data.train import TrainDataset


class CandidateEncoder(tf.keras.layers.Layer):
    def __init__(self, unique_movie_ids, embedding_dimension):
        super().__init__()
        self._embedding_movie_ids = tf.keras.Sequential([
            tf.keras.layers.IntegerLookup(vocabulary=unique_movie_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension),
        ])

        self._gru_encoder = tf.keras.layers.GRU(units=embedding_dimension, recurrent_initializer="glorot_uniform")

    def call(self, features):
        seq_embedding_movie_ids = self._embedding_movie_ids(features["previous_movie_ids"])

        encoder = self._gru_encoder(seq_embedding_movie_ids)

        return encoder


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
