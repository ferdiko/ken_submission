import logging


LOGGING_LEVEL = logging.DEBUG

def setup_logger():
    formatter = logging.Formatter(fmt='%(asctime)s - %(module)s:  %(message)s')
    logger = logging.getLogger("es")

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.setLevel(LOGGING_LEVEL)
        logger.addHandler(handler)
    return logger

