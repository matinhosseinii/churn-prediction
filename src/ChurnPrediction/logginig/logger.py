import logging
import sys
import os
from logging.handlers import RotatingFileHandler

# Define the format for the log messages
LOG_FORMAT = "%(filename)s:%(lineno)d [ %(asctime)s ] %(levelname)s - %(message)s"

# Function to set up a configured logger.
def setup_logger(name="churn-prediction",
                 log_file=os.path.join("logs", "churn-prediction.log"),
                 level=logging.INFO):
    
    try:
        # Get the directory part of the log file path
        log_dir = os.path.dirname(log_file)
        # Create the directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating log directory: {e}")

    logger = logging.getLogger(name)
    
    # 🛡️ Prevents adding handlers multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)

    # ⚙️ Handler for rotating log files (e.g., max 5MB, keep 5 old files)
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    
    # Handler for console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger