import yaml
import datetime


class Configuration:
    def __init__(self, filename: str):

        with open(filename, "r") as f:
            conf = yaml.safe_load(f)

        self._general = conf.get("general")
        self._data_preprocessing = conf.get('data_preprocessing')
        self.project = self._general.get("project")
        self.region = self._general.get("region")
        self.model_name = self._general.get("model_name")
        self.bucket_name = self._general.get("bucket_name")
        self.dataset_id = self._data_preprocessing.get("dataset_id")
        self.output_filename = self._data_preprocessing.get("output_filename")

        self._timestamp = None
        self.timestamp = self._general.get("timestamp", None)

        self.artifact_store = f"gs://{self.model_name}-kfp-artifact-store"
        self.pipeline_root = f"{self.artifact_store}/{self.timestamp}/pipeline"
        self.data_root = f"{self.artifact_store}/{self.timestamp}/data"
        self.base_output_dir = f"{self.artifact_store}/{self.timestamp}/models"

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        if value is None:
            value = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self._timestamp = value
