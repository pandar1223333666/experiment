"""
项目自定义日志
"""

import logging

def setup_logging():
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger("app")
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        logger.addHandler(stream_handler)

    logger.propagate = False

def get_logger(name):
    return logging.getLogger(f"app.{name}")
