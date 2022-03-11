from typing import NamedTuple
from kfp.v2.dsl import Dataset, Output, component
from pipeline.kfp_components.dependencies import PYTHON37, GOOGLE_CLOUD_BIGQUERY, TENSORFLOW, PANDAS


@component(base_image=PYTHON37, packages_to_install=[GOOGLE_CLOUD_BIGQUERY, TENSORFLOW, PANDAS])
def base_preprocessing(
    project_id: str,
    data_root: str,
    dataset_id: str,
    output_filename: str,
    seq_length: int = 10,
    data_base_path: str,
    output_filename: str


):

    import os
    from google.cloud import bigquery
    import pandas as pd
    import tensorflow as tf

    output_path = os.path.join(data_base_path, output_filename)

    def load_dataset(query) -> pd.DataFrame:
        bq_client = bigquery.Client(project=project_id)
        results = bq_client.query(query).to_dataframe()
        return results

    def read_tf_dataset(output_path) -> tf.data.Dataset:
        return tf.data.experimental.load(output_path)

    def save_tf_dataset(dict_features) -> None:
        dataset = tf.data.Dataset.from_tensor_slices(dict_features)

        if not tf.io.gfile.exists(data_base_path):
            tf.io.gfile.makedirs(data_base_path)

        tf.data.experimental.save(dataset, output_path)