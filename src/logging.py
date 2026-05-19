from __future__ import absolute_import

import logging

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.handlers = []

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file, mode='w', encoding="utf-8")
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger