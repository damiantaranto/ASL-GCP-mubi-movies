from kfp.v2.dsl import Dataset, Output, component
from pipeline.kfp_components.dependencies import PYTHON37, GOOGLE_CLOUD_BIGQUERY, TENSORFLOW, PANDAS, PYARROW


@component(base_image=PYTHON37, packages_to_install=[GOOGLE_CLOUD_BIGQUERY, TENSORFLOW, PANDAS, PYARROW])
def inference_dataset(
        project_id: str,
        data_root: str,
        movies_output_filename: str,
):
    import os
    from google.cloud import bigquery
    import pandas as pd
    import tensorflow as tf

    name_transformation = "inference"

    sql_query = f"""WITH ratings AS (
        SELECT
            ratings.user_id,
            ratings.movie_id,
            ratings.rating_id,
            ratings.rating_timestamp_utc,
            ratings.rating_score,
            COALESCE(ratings.user_eligible_for_trial, False) AS user_eligible_for_trial,
            COALESCE(ratings.user_has_payment_method, False) AS user_has_payment_method,
            COALESCE(ratings.user_subscriber, False) AS user_subscriber,
            COALESCE(ratings.user_trialist, False) AS user_trialist,
            movies.movie_title,
            COALESCE(ratings.rating_score,
             PERCENTILE_DISC(ratings.rating_score, 0.5) OVER (PARTITION BY ratings.movie_id)) AS rating_value,
            COALESCE(DATE_DIFF(ratings.rating_timestamp_utc, LAG(ratings.rating_timestamp_utc, 1)
                OVER (PARTITION BY user_id ORDER BY ratings.rating_timestamp_utc),  DAY), -1) AS days_since_last_rating,
            COALESCE(movies.movie_release_year, 0) as movie_release_year,
            movies.movie_title_language,
            LAST_VALUE(ratings.rating_timestamp_utc) OVER (PARTITION BY user_id
                ORDER BY ratings.rating_timestamp_utc
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_rating_timestamp
        FROM `__DATASET_ID__.mubi_ratings_data` ratings
        JOIN `__DATASET_ID__.mubi_movie_data` movies ON
            ratings.movie_id = movies.movie_id
            AND ratings.rating_score IS NOT NULL
    ),
    shifted_last_rating AS (
         SELECT
             ratings.user_id,
             ratings.movie_id,
             COALESCE(LAG(ratings.days_since_last_rating, 1) over (PARTITION BY user_id
                ORDER BY ratings.days_since_last_rating DESC), 0) as shifted_days,
         FROM ratings
    ),
    sequenced_rating AS (
        SELECT
            movie_title,
            ratings.movie_id,
            ratings.user_id,
            rating_value,
            movie_release_year,
            rating_timestamp_utc,
            user_eligible_for_trial,
            days_since_last_rating,
            user_has_payment_method,
            user_subscriber,
            user_trialist,
            last_rating_timestamp,
            ARRAY_AGG(ratings.movie_id) OVER (PARTITION BY ratings.user_id
                ORDER BY rating_timestamp_utc asc
                ROWS BETWEEN __FRAME_START__ PRECEDING AND __FRAME_END__ PRECEDING) AS previous_movie_ids,
            ARRAY_AGG(movie_release_year) OVER (PARTITION BY ratings.user_id
                ORDER BY rating_timestamp_utc asc
                ROWS BETWEEN __FRAME_START__ PRECEDING AND __FRAME_END__ PRECEDING) AS previous_movie_years,
            ARRAY_AGG(rating_value) OVER (PARTITION BY ratings.user_id
                ORDER BY rating_timestamp_utc asc
                ROWS BETWEEN __FRAME_START__ PRECEDING AND __FRAME_END__ PRECEDING) AS previous_score,
            ARRAY_AGG(shifted_days) OVER (PARTITION BY ratings.user_id
                ORDER BY rating_timestamp_utc asc
                ROWS BETWEEN __FRAME_START__ PRECEDING AND __FRAME_END__ PRECEDING) AS previous_days_since_last_rating,
        FROM ratings
        INNER JOIN shifted_last_rating ON
            ratings.user_id = shifted_last_rating.user_id
            AND ratings.movie_id = shifted_last_rating.movie_id
    )
    SELECT * FROM sequenced_rating
    WHERE ARRAY_LENGTH(previous_movie_ids) > 2
        AND rating_timestamp_utc = last_rating_timestamp"""


    data_path = os.path.join(data_root, name_transformation)

    output_path = os.path.join(data_path, movies_output_filename)

    def load_dataset(query) -> pd.DataFrame:
        bq_client = bigquery.Client()
        results = bq_client.query(query, project=project_id).to_dataframe()
        return results

    def save_tf_dataset(dict_features) -> None:
        dataset = tf.data.Dataset.from_tensor_slices(dict_features)

        if not tf.io.gfile.exists(data_path):
            tf.io.gfile.makedirs(data_path)

        tf.data.experimental.save(dataset, output_path)

    def get_features_dict(self, rows) -> dict:
        dict_features = dict(
            **rows[["user_id"]].astype("int"),
            **rows[["user_eligible_for_trial"]].astype("int"),
            **rows[["user_has_payment_method"]].astype("int"),
            **rows[["user_subscriber"]].astype("int"),
            **rows[["user_trialist"]].astype("int"),
            **{"previous_movie_ids":
                tf.keras.preprocessing.sequence.pad_sequences(rows["previous_movie_ids"].values,
                                                              maxlen=self.seq_length, dtype='int32', value=0)},
            **{"previous_movie_years":
                tf.keras.preprocessing.sequence.pad_sequences(rows["previous_movie_years"].values,
                                                              maxlen=self.seq_length, dtype='float32', value=1980.0)},
            **{"previous_score":
                tf.keras.preprocessing.sequence.pad_sequences(rows["previous_score"].values,
                                                              maxlen=self.seq_length, dtype='float32', value=2.5)},
            **{"previous_days_since_last_rating":
                tf.keras.preprocessing.sequence.pad_sequences(rows["previous_days_since_last_rating"].values,
                                                              maxlen=self.seq_length, dtype='float32', value=0)}
        )
        return dict_features

    try:
        assert not tf.io.gfile.exists(output_path)
    except AssertionError:
        raise ValueError("Dataset already exists, load it. Remove timestamp from config in case of new run")

    sql_query = (
        sql_query
            .replace("__DATASET_ID__", str("mubi_movie_data"))
            .replace("__FRAME_START__", str("9"))
            .replace("__FRAME_END__", str("0"))
    )

    rows = load_dataset(sql_query)

    dict_features = get_features_dict(rows)
    del rows

    save_tf_dataset(dict_features)
