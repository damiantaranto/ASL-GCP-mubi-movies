from kfp.v2.dsl import Dataset, Output, component
from pipeline.kfp_components.dependencies import PYTHON37, GOOGLE_CLOUD_BIGQUERY, TENSORFLOW, PANDAS


@component(base_image=PYTHON37, packages_to_install=[GOOGLE_CLOUD_BIGQUERY, TENSORFLOW, PANDAS])
def movies_dataset(
        project_id: str,
        data_root: str,
        movies_output_filename: str,
):
    import os
    from google.cloud import bigquery
    import pandas as pd
    import tensorflow as tf

    name_transformation = "movies"

    sql_query = f"""SELECT DISTINCT
            ratings.movie_id,
            movies.movie_title,
        FROM `raw_dataset.mubi_ratings_data` ratings
        JOIN `raw_dataset.mubi_movie_data` movies ON
            ratings.movie_id = movies.movie_id"""

    data_path = os.path.join(data_root, name_transformation)

    output_path = os.path.join(data_path, movies_output_filename)

    def load_dataset(query) -> pd.DataFrame:
        bq_client = bigquery.Client(project=project_id)
        results = bq_client.query(query).to_dataframe()
        return results

    def save_tf_dataset(dict_features) -> None:
        dataset = tf.data.Dataset.from_tensor_slices(dict_features)

        if not tf.io.gfile.exists(data_path):
            tf.io.gfile.makedirs(data_path)

        tf.data.experimental.save(dataset, output_path)

    def get_features_dict(rows) -> dict:
        dict_features = dict(
            **rows[["movie_id"]].astype("int"),
            **rows[["movie_title"]].astype("str"),
        )
        return dict_features

    try:
        assert not tf.io.gfile.exists(output_path)
    except AssertionError:
        raise ValueError("Dataset already exists, load it. Remove timestamp from config in case of new run")

    rows = load_dataset(sql_query)

    dict_features = get_features_dict(rows)
    del rows

    save_tf_dataset(dict_features)
