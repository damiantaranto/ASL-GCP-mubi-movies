import os
from abc import ABC, abstractmethod

import pandas as pd
import tensorflow as tf

from google.cloud import bigquery


class AbstractDataPrep(ABC):
    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_data_path(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def set_output_filename(self) -> str:
        raise NotImplementedError


class DataPrep(AbstractDataPrep):
    def __init__(self, configuration, **kwargs):
        self.configuration = configuration
        self._project = configuration.project
        self.data_root = configuration.data_root
        self.dataset_id = configuration.dataset_id
        self.output_filename = configuration.output_filename
        self.seq_length = kwargs.get("seq_length", 10)
        self._data_base_path = self.set_data_path()
        self._output_filename = self.set_output_filename()
        self.output_path = os.path.join(self._data_base_path, self._output_filename)

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
