'''Sets up console logging for all levels of log outputs.'''
import logging
import os

# Root logger
def configure_logging(name="vllm_simulator", log_to_file=False, log_file_path="outputs/logs/output.log"):
    """
    Configures the logging setup.

    Args:
        name (str): Name of the logger.
        log_to_file (bool): Whether to write logs to a file.
        log_file_path (str): Path to the log file if log_to_file is True.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Check if the logger already has handlers to avoid adding multiple handlers
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)  # Set the logging level for the logger

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create a formatter and attach it to the console handler
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
        console_handler.setFormatter(formatter)

        # Add the console handler to the logger
        logger.addHandler(console_handler)

        # Optionally add a file handler if log_to_file is True
        if log_to_file:
            # Ensure the directory for the log file exists
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

            # Create a file handler
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.DEBUG)

            # Use the same formatter for the file handler
            file_handler.setFormatter(formatter)

            # Add the file handler to the logger
            logger.addHandler(file_handler)

    # If you don't want child loggers to propagate to the parent
    logger.propagate = False

    return logger
