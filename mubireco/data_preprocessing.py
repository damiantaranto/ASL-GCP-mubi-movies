# Stdlib
import argparse
import time
import logging

# app libraries
from mubireco.utils.configuration import Configuration
from mubireco.utils.logging import get_or_create_logger

from mubireco.data.train import TrainDataset
from mubireco.data.inference import InferenceDataset
from mubireco.data.movies import MoviesDataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_length", help="Maximum sequence length", type=int, default=10)
    parser.add_argument("--log", "-l", help="set the logging level", type=str, default="INFO")

    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log}")

    logger = get_or_create_logger(numeric_level)

    config = Configuration("mubireco/config/config.yaml")

    logger.info(f"Timestamp: {config.timestamp}")

    start = time.time()
    logger.info("Train dataset creation")
    TrainDataset(config, seq_length=args.seq_length).run()

    logger.info("Inference dataset creation")
    InferenceDataset(config, seq_length=args.seq_length).run()

    logger.info("Movies dataset creation")
    MoviesDataset(config).run()

    logger.info(f"Total execution time: {(time.time() - start) / 60:,.0f} mins")
