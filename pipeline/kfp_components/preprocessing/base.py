from typing import NamedTuple
from kfp.v2.dsl import Dataset, Output, component
from pipeline.kfp_components.dependencies import PYTHON37, GOOGLE_CLOUD_BIGQUERY, TENSORFLOW, PANDAS


@component(base_image=PYTHON37, packages_to_install=[GOOGLE_CLOUD_BIGQUERY, TENSORFLOW, PANDAS])
def base_preprocessing(
    project: str,
    data_root: str,
    dataset_id: str,
    output_filename: str,
    seq_length: int = 10,
    _data_base_path: str = set_data_path(),
    file_pattern: str = None,
):

    import logging
    import os
    from google.cloud.exceptions import GoogleCloudError
    from google.cloud import bigquery
    import pandas as pd
    import tensorflow as tf

    def load_dataset(self, query) -> pd.DataFrame:
        bq_client = bigquery.Client(project=self._project)

        results = bq_client.query(query).to_dataframe()

        return results

    def read_tf_dataset(self) -> tf.data.Dataset:
        return tf.data.experimental.load(self.output_path)

    def save_tf_dataset(self, dict_features) -> None:
        dataset = tf.data.Dataset.from_tensor_slices(dict_features)

        if not tf.io.gfile.exists(self._data_base_path):
            tf.io.gfile.makedirs(self._data_base_path)

        tf.data.experimental.save(dataset, self.output_path)