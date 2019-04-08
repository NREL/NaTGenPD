# -*- coding: utf-8 -*-
"""
Command Line Interface and Entry point
@author: mrossol
"""
# import click
import logging
# import os

FORMAT = '%(levelname)s - %(asctime)s [%(filename)s:%(lineno)d] : %(message)s'
LOG_LEVEL = {'INFO': logging.INFO,
             'DEBUG': logging.DEBUG,
             'WARNING': logging.WARNING,
             'ERROR': logging.ERROR,
             'CRITICAL': logging.CRITICAL}


def get_handler(log_level="INFO", log_file=None, log_format=FORMAT):
    """
    get logger handler

    Parameters
    ----------
    log_level : str
        handler-specific logging level, must be key in LOG_LEVEL.
    log_file : str
        path to the log file
    log_format : str
        format string to use with the logging package

    Returns
    -------
    handler : logging.FileHandler | logging.StreamHandler
        handler to add to logger
    """
    if log_file:
        # file handler with mode "a"
        handler = logging.FileHandler(log_file, mode='a')
    else:
        # stream handler to system stdout
        handler = logging.StreamHandler()

    if log_format:
        logformat = logging.Formatter(log_format)
        handler.setFormatter(logformat)

    # Set a handler-specific logging level (root logger should be at debug)
    handler.setLevel(LOG_LEVEL[log_level])

    return handler


def setup_logger(logger_name, log_level="INFO", log_file=None,
                 log_format=FORMAT):
    """
    Setup logging instance with given name and attributes

    Parameters
    ----------
    logger_name : str
        Name of logger
    log_level : str
        Level of logging to capture, must be key in LOG_LEVEL. If multiple
        handlers/log_files are requested in a single call of this function,
        the specified logging level will be applied to all requested handlers.
    log_file : str | list
        Path to file to use for logging, if None use a StreamHandler
        list of multiple handlers is permitted
    log_format : str
        Format for loggings, default is FORMAT

    Returns
    -------
    logger : logging.logger
        instance of logger for given name, with given level and added handler
    handler : logging.FileHandler | logging.StreamHandler | list
        handler(s) added to logger
    """
    logger = logging.getLogger(logger_name)
    current_handlers = [str(h) for h in logger.handlers]

    # Set root logger to debug, handlers will control levels above debug
    logger.setLevel(LOG_LEVEL["DEBUG"])

    handlers = []
    if isinstance(log_file, list):
        for h in log_file:
            handlers.append(get_handler(log_level=log_level, log_file=h,
                                        log_format=log_format))
    else:
        handlers.append(get_handler(log_level=log_level, log_file=log_file,
                                    log_format=log_format))
    for handler in handlers:
        if str(handler) not in current_handlers:
            logger.addHandler(handler)
    return logger
