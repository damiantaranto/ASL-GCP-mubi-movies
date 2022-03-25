import logging


def get_or_create_logger(numeric_level: int = None):
    """
    Creates a project-wide logger at the given logging level.
    :param numeric_level: the logging level to set (eg 20 for INFO).
    :return: Logger singleton.
    """

    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=numeric_level)

    return logging.getLogger("MubiReco")
