from kfp.v2.dsl import Dataset, Output, component
from pipeline.kfp_components.model_dependencies import PYTHON37, GOOGLE_CLOUD_BIGQUERY, TENSORFLOW, NUMPY, TENSORFLOW_RECOMMENDERS, PYARROW

# gcr.io/qwiklabs-gcp-04-424f1fdacc59/moviecust@sha256:52ec0a72e8972601e00469561a69b22c07d6a250d25107f86d16fbdd86eb3b93

@component(base_image=moviecust:0.1, packages_to_install=[GOOGLE_CLOUD_BIGQUERY, TENSORFLOW, NUMPY, TENSORFLOW_RECOMMENDERS, PYARROW])
def train_dataset(
        project_id: str,
        data_root: str,
        batch_size: int,
        embedding_dimension: int,
        filename: str,
        #seq_length: int,
):
    import os
    from datetime import datetime
    from google.cloud import bigquery
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import tensorflow_recommenders as tfrs

    from mubireco.data.movies import MoviesDataset
    from mubireco.data.train import TrainDataset    
    from mubireco.mode.model_definition import MubiMoviesModel, CandidateEncoder

    region = os.getenv("VERTEX_LOCATION")
    model_name = "ml_movie_user_recommender"
    #self.dataset_id = self._data_preprocessing.get("dataset_id")
    #self.output_filename = self._data_preprocessing.get("output_filename")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    artifact_store = f"gs://{project_id}-kfp-artifact-store"
    pipeline_root = f"{self.artifact_store}/{timestamp}/pipeline"
    data_root = f"{self.artifact_store}/{timestamp}/data"
    base_output_dir = f"{self.artifact_store}/{self.timestamp}/models"
    
    def create_model(batch_size, embedding_dimension, config) -> MubiMoviesModel:
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


    def load_dataset(query) -> pd.DataFrame:
        bq_client = bigquery.Client()
        results = bq_client.query(query, project=project_id).to_dataframe()
        return results

    def save_tf_dataset(dict_features) -> None:
        dataset = tf.data.Dataset.from_tensor_slices(dict_features)

        if not tf.io.gfile.exists(data_path):
            tf.io.gfile.makedirs(data_path)

        tf.data.experimental.save(dataset, output_path)

    def get_features_dict(rows) -> dict:
        dict_features = dict(
            **rows[["movie_id"]].astype("int"),
            **rows[["user_eligible_for_trial"]].astype("int"),
            **rows[["user_has_payment_method"]].astype("int"),
            **rows[["user_subscriber"]].astype("int"),
            **rows[["user_trialist"]].astype("int"),
            **{"previous_movie_ids":
                tf.keras.preprocessing.sequence.pad_sequences(rows["previous_movie_ids"].values,
                                                              maxlen=seq_length, dtype='int32', value=0)},
            **{"previous_movie_years":
                tf.keras.preprocessing.sequence.pad_sequences(rows["previous_movie_years"].values,
                                                              maxlen=seq_length, dtype='float32', value=1980.0)},
            **{"previous_score":
                tf.keras.preprocessing.sequence.pad_sequences(rows["previous_score"].values,
                                                              maxlen=seq_length, dtype='float32', value=2.5)},
            **{"previous_days_since_last_rating":
                tf.keras.preprocessing.sequence.pad_sequences(rows["previous_days_since_last_rating"].values,
                                                              maxlen=seq_length, dtype='float32', value=0)}
        )
        return dict_features

    try:
        assert not tf.io.gfile.exists(output_path)
    except AssertionError:
        raise ValueError("Dataset already exists, load it. Remove timestamp from config in case of new run")

    sql_query = (
        sql_query
            .replace("__DATASET_ID__", str("mubi_movie_data"))
            .replace("__FRAME_START__", str(10))
            .replace("__FRAME_END__", str(1))
    )

    rows = load_dataset(sql_query)

    dict_features = get_features_dict(rows)
    del rows

    save_tf_dataset(dict_features)
