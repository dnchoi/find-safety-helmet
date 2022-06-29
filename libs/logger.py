import logging
import os
import sys
from logging import handlers

import colorlog


class Logger:
    def __init__(self, className, filePath=None):
        self.className = className
        if filePath is None:
            self.filePath = "./log/"
        if not os.path.exists(self.filePath):
            os.makedirs(self.filePath)

    def logger_LUT(self, idx):
        logger_Level = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        return logger_Level.get(idx, "INFO")

    def initLogger(self, idx, logger_name):
        __logger = logging.getLogger(self.className)
        if len(__logger.handlers) > 0:
            return __logger
        streamFormatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(levelname)-8s]%(reset)s <%(name)s>: %(module)s:%(lineno)d:  %(message)s"
        )
        fileFormatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] <%(name)s>: %(module)s:%(lineno)d: %(message)s"
        )

        streamHandler = colorlog.StreamHandler(sys.stdout)

        fileHandler = handlers.TimedRotatingFileHandler(
            os.path.abspath("{}/{}.log".format(self.filePath, logger_name)),
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
        )
        streamHandler.setFormatter(streamFormatter)
        fileHandler.setFormatter(fileFormatter)

        __logger.addHandler(streamHandler)
        __logger.addHandler(fileHandler)

        __logger.setLevel(self.logger_LUT(idx))

        return __logger
