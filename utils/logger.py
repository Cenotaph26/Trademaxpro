"""
Logger setup — propagate=True ile root logger'a ulaşır (log buffer için).
"""
import logging
import sys


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    # Handler ekleme — root logger'daki handler'lar propagate ile çalışır
    logger.propagate = True
    # Level ayarla ama handler ekleme (root'ta var)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger
