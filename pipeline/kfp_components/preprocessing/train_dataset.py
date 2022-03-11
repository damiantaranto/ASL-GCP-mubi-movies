from kfp.v2.dsl import Dataset, Output, component
from pipeline.kfp_components.dependencies import PYTHON37, GOOGLE_CLOUD_BIGQUERY, TENSORFLOW, PANDAS

@component(base_image=PYTHON37, packages_to_install=[GOOGLE_CLOUD_BIGQUERY, TENSORFLOW, PANDAS])
def train_dataset(
    project_id: str,
    data_root: str,
    dataset_id: str,
    output_filename: str,
    seq_length: int = 10,

    # self._project = configuration.project
    # self.data_root = configuration.data_root
    # self.dataset_id = configuration.dataset_id
    # self.output_filename = configuration.output_filename
    # self.seq_length = kwargs.get("seq_length", 10)
    # self._data_base_path = self.set_data_path()
    # self._output_filename = self.set_output_filename()
    # self.output_path = os.path.join(self._data_base_path, self._output_filename)
):
    import os
    from google.cloud import bigquery
    import pandas as pd
    import tensorflow as tf

    name_transformation = "train"

    sql_query = f"""WITH ratings AS (
           SELECT
               ratings.user_id,
               ratings.movie_id,
               ratings.rating_id,
               ratings.rating_timestamp_utc,
               ratings.rating_score,
               COALESCE(ratings.user_eligible_for_trial, False) as user_eligible_for_trial,
               COALESCE(ratings.user_has_payment_method, False) as user_has_payment_method,
               COALESCE(ratings.user_subscriber, False) as user_subscriber,
               COALESCE(ratings.user_trialist, False) as user_trialist,
               movies.movie_title,
               COALESCE(movies.movie_release_year, 0) as movie_release_year,
               movies.movie_title_language,
               LAST_VALUE(ratings.rating_timestamp_utc) OVER (PARTITION BY user_id
                   ORDER BY ratings.rating_timestamp_utc ASC
                   ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_rating_timestamp
           FROM `__DATASET_ID__.mubi_ratings_data` ratings
           JOIN `__DATASET_ID__.mubi_movie_data` movies ON
               ratings.movie_id = movies.movie_id
       ), sequenced_rating AS (
           SELECT
               movie_title,
               movie_id,
               user_id,
               user_eligible_for_trial,
               user_has_payment_method,
               user_subscriber,
               user_trialist,
               (ARRAY_AGG(movie_id) OVER (PARTITION BY user_id
                   ORDER BY rating_timestamp_utc
                   ROWS BETWEEN __FRAME_START__ PRECEDING AND __FRAME_END__ PRECEDING)) AS previous_movie_ratings,
               (ARRAY_AGG(movie_release_year) OVER (PARTITION BY user_id
                   ORDER BY rating_timestamp_utc
                   ROWS BETWEEN __FRAME_START__ PRECEDING AND __FRAME_END__ PRECEDING)) AS previous_movie_years
           FROM ratings
       )
       SELECT * FROM sequenced_rating"""

    data_path = os.path.join(data_root, name_transformation)

    output_filename = output_filename.get(name_transformation)

    output_path = os.path.join(data_path, output_filename)

    def load_dataset(query) -> pd.DataFrame:
        bq_client = bigquery.Client(project=project_id)
        results = bq_client.query(query).to_dataframe()
        return results

    def read_tf_dataset(output_path) -> tf.data.Dataset:
        return tf.data.experimental.load(output_path)

    def save_tf_dataset(dict_features) -> None:
        dataset = tf.data.Dataset.from_tensor_slices(dict_features)

        if not tf.io.gfile.exists(data_path):
            tf.io.gfile.makedirs(data_path)

        tf.data.experimental.save(dataset, output_path)


    def get_features_dict(self, rows) -> dict:
        dict_features = dict(
            **rows[["movie_id"]].astype("int"),
            **rows[["user_eligible_for_trial"]].astype("int"),
            **rows[["user_has_payment_method"]].astype("int"),
            **rows[["user_subscriber"]].astype("int"),
            **rows[["user_trialist"]].astype("int"),
            **{"previous_movie_ratings":
                   tf.keras.preprocessing.sequence.pad_sequences(rows["previous_movie_ratings"].values,
                                                                 maxlen=self.seq_length, dtype='int32', value=0)},
            **{"previous_movie_years":
                   tf.keras.preprocessing.sequence.pad_sequences(rows["previous_movie_years"].values,
                                                                 maxlen=self.seq_length, dtype='float32', value=1980.0)}
        )
        return dict_features


    try:
        assert not tf.io.gfile.exists(output_path)
    except AssertionError:
        raise ValueError("Dataset already exists, load it. Remove timestamp from config in case of new run")

    query = (
        sql_query
            .replace("__DATASET_ID__", str(dataset_id))
            .replace("__FRAME_START__", str(start_frame))
            .replace("__FRAME_END__", str(end_frame))
    )

    rows = load_dataset(query)

    dict_features = get_features_dict(rows)
    del rows

    return save_tf_dataset(dict_features)