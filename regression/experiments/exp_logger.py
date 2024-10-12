import logging
import os

# Function to create a logger that saves logs to a specified directory
def get_logger(logger_name='exp_logger', log_dir='logs', log_file='exp_log.log'):
    # Ensure the log directory exists, create it if it doesn't
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Full log file path (directory + log file name)
    log_file_path = os.path.join(log_dir, log_file)

    # Create or get the logger
    logger = logging.getLogger(logger_name)

    # Prevent adding multiple handlers if logger already exists
    if not logger.hasHandlers():
        # Set the log level
        logger.setLevel(logging.INFO)

        # Create a file handler to log to the specified file
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        # Create a console handler to also log to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a log formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Attach formatter to handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
