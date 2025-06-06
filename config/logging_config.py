import logging
import os

def setup_logging():
    """
    Sets up the logging configuration for the application.

    Logs will be directed to both the console and a file named 'app.log'.
    The log level for the console is INFO, and for the file is DEBUG.
    """
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_path = os.path.join(log_directory, "app.log")

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Overall lowest level

    # Clear existing handlers to prevent duplicate logs in Streamlit
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logging.info("Logging setup complete.")

if __name__ == "__main__":
    setup_logging()
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")
