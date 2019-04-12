# -*- coding: utf-8 -*-
"""
Command Line Interface and Entry point
@author: mrossol
"""
import click
import logging
import os
from .clean import ParseSmoke, CleanSmoke
from .handler import CEMS

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


@click.group()
@click.option('--log_file', default=None, type=click.Path(),
              help='Path to .log file')
@click.option('--verbose', '-v', is_flag=True,
              help='If used upgrade logging to DEBUG')
@click.pass_context
def main(ctx, log_file, verbose):
    """
    CLI entry point
    """
    ctx.ensure_object(dict)

    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    ctx.obj['LOG_LEVEL'] = log_level
    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    ctx.obj['LOG_FILE'] = log_file


@main.command()
@click.argument('smoke_dir', type=click.Path(exists=True))
@click.argument('year', type=int)
@click.pass_context
def import_smoke_data(ctx, smoke_dir, year):
    """
    Parse Smoke data from .txt files in 'smoke_dir',
    extract performance variables,
    and save to disc as a .h5 file

    Parameters
    ----------
    smoke_dir : str
        Path to root directory containing SMOKE .txt files
    year : int | str
        Year to parse
    """
    setup_logger("NaTGenPD.precleaning", log_file=ctx.obj['LOG_FILE'],
                 log_level=ctx.obj['LOG_LEVEL'])
    ParseSmoke.performance_vars(smoke_dir, year, save=True)


@main.command()
@click.argument('smoke_file', type=click.Path(exists=True))
@click.argument('unit_attrs', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--cc_map', '-cc', type=click.Path(exists=True),
              help='Path to .csv with CEMS to EIA CC unit mapping')
@click.pass_context
def clean_smoke_data(ctx, smoke_file, unit_attrs, out_file, cc_map):
    """
    Clean-up SMOKE data for heat rate analysis:
    - Convert gross load to net load
    - Remove null/unrealistic values
    - Remove start-up and shut-down

    Parameters
    ----------
    smoke_file : str
        Path to .h5 file with SMOKE data
    unit_attrs : str
        Path to .csv containing facility (unit) attributes
    out_file : str
        Path to output .h5 file to write clean-data too
    """
    setup_logger("NaTGenPD.precleaning", log_file=ctx.obj['LOG_FILE'],
                 log_level=ctx.obj['LOG_LEVEL'])
    CleanSmoke.clean(smoke_file, unit_attrs_path=unit_attrs, cc_map=cc_map,
                     out_file=out_file)


@main.command()
@click.argument('comp_file', type=click.Path())
@click.argument('year_files', type=click.Path(exists=True), nargs=-1)
@click.pass_context
def combine_clean_files(ctx, comb_file, year_files):
    """
    Combine multiple years of CEMS data into a single file

    Parameters
    ----------
    comb_file : str
        Path to .h5 file to combine data into
    year_files : list | tuple
        List of file paths to combine.
        Each file should correspond to a years worth of Clean SMOKE data
    """
    setup_logger("NaTGenPD.handler", log_file=ctx.obj['LOG_FILE'],
                 log_level=ctx.obj['LOG_LEVEL'])
    CEMS.combine_years(comb_file, year_files)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        raise
