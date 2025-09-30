import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def get_logger(name: str = "face_recognition", log_file: str = "logs/pipeline.log") -> logging.Logger:
    """
    Create and return a logger with console + rotating file handlers.
    Ensures the log folder exists and uses UTF-8 encoding.
    """
    # Ensure folder exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # File handler (rotating, keeps last 5 files of 1MB each)
    fh = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Add handlers
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
