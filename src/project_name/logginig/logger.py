# import logging
# import os

# def get_logger(name: str = __name__, log_level: int = logging.INFO) -> logging.Logger:
#     """
#     Initializes and returns a logger with the specified name and log level.
#     Logs are written to both console and a file named 'app.log' in the current directory.
#     """
#     logger = logging.getLogger(name)
#     logger.setLevel(log_level)
#     logger.propagate = False

#     if not logger.handlers:
#         # Console handler
#         console_handler = logging.StreamHandler()
#         console_handler.setLevel(log_level)
#         console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         console_handler.setFormatter(console_format)
#         logger.addHandler(console_handler)

#         # File handler
#         log_file = os.path.join(os.path.dirname(__file__), 'app.log')
#         file_handler = logging.FileHandler(log_file)
#         file_handler.setLevel(log_level)
#         file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         file_handler.setFormatter(file_format)
#         logger.addHandler(file_handler)

#     return logger