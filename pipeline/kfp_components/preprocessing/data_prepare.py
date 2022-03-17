from kfp.v2.dsl import Artifact, Input, Output, Dataset, Model, component
from pipeline.kfp_components.dependencies import PYTHON37, TENSORFLOW, TENSORFLOW_RECOMMENDERS, PANDAS, GOOGLE_CLOUD_BIGQUERY, PYARROW, GOOGLE_CLOUD_STORAGE

@component(base_image="gcr.io/qwiklabs-gcp-04-424f1fdacc59/moviecust:0.1", packages_to_install=[TENSORFLOW, TENSORFLOW_RECOMMENDERS, PANDAS, GOOGLE_CLOUD_BIGQUERY, PYARROW, GOOGLE_CLOUD_STORAGE])
def data_prepare(
):
    import argparse
    import time
    import logging

    # app libraries
    from mubireco.utils.configuration import Configuration
    from mubireco.utils.logging import get_or_create_logger

    from mubireco.data.train import TrainDataset
    from mubireco.data.inference import InferenceDataset
    from mubireco.data.movies import MoviesDataset

    config = Configuration("mubireco/config/config.yaml")

    start = time.time()
    # logger.info("Train dataset creation")
    # TrainDataset(config, seq_length=10).run()

    # logger.info("Inference dataset creation")
    InferenceDataset(config, seq_length=10).run()

    # logger.info("Movies dataset creation")
    MoviesDataset(config).run()

    # logger.info(f"Total execution time: {(time.time() - start) / 60:,.0f} mins")