"""
HPC Script to run CEMS procedure analysis
"""
import os
import sys
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)
import NaTGenPD as npd
from NaTGenPD.analysis import ProcedureAnalysis, QuartileAnalysis


def process(data_dir, out_dir, log_file):
    """
    Parse, clean, filter, and fit all units for the given years

    Parameters
    ----------
    data_dir : str
        Directory containing SMOKE data
    out_dir : str
        Directory into which output analysis is to be saved
    log_file : str
        Path to file to use for logging
    """
    npd.setup_logger('NaTGenPD', log_level='INFO', log_file=log_file)
    fits_dir = os.path.join(data_dir, 'Final_Fits')
    raw_paths = [os.path.join(data_dir, '{y}/SMOKE_{y}.h5'.format(y=y))
                 for y in (2016, 2017)]
    clean_path = os.path.join(data_dir, 'SMOKE_Clean_2016-2017.h5')
    filter_path = os.path.join(data_dir, 'SMOKE_Filtered_2016-2017.h5')
    process_dir = os.path.join(out_dir, 'process')
    if not os.path.exists(process_dir):
        os.makedirs(process_dir)

    out_path = os.path.join(process_dir, 'process_stats_2016-2017.csv')

    ProcedureAnalysis.stats(fits_dir, raw_paths, clean_path, filter_path,
                            out_path)


def quartiles(data_dir, out_dir, log_file):
    """
    Parse, clean, filter, and fit all units for the given years

    Parameters
    ----------
    data_dir : str
        Directory containing SMOKE data
    out_dir : str
        Directory into which output analysis is to be saved
    log_file : str
        Path to file to use for logging
    """
    logger = npd.setup_logger('NaTGenPD', log_level='INFO', log_file=log_file)

    logger.info('Running Quartile Analysis for Filtered Units')
    fits_dir = os.path.join(data_dir, 'CEMS_Fits')
    filter_path = os.path.join(data_dir, 'SMOKE_Filtered_2016-2017.h5')
    quartile_dir = os.path.join(out_dir, 'filtered_fits')
    if not os.path.exists(quartile_dir):
        os.makedirs(quartile_dir)

    out_path = os.path.join(quartile_dir, 'filtered_quartile_stats.csv')

    QuartileAnalysis.stats(fits_dir, filter_path, out_path)

    logger.info('Running Quartile Analysis for Final Units')
    fits_dir = os.path.join(data_dir, 'Final_Fits')
    filter_path = os.path.join(data_dir, 'SMOKE_Filtered_2016-2017.h5')
    quartile_dir = os.path.join(out_dir, 'final_fits')
    if not os.path.exists(quartile_dir):
        os.makedirs(quartile_dir)

    out_path = os.path.join(quartile_dir, 'final_quartile_stats.csv')

    QuartileAnalysis.stats(fits_dir, filter_path, out_path)


if __name__ == '__main__':
    data_dir = '/scratch/mrossol/CEMS'
    out_dir = os.path.join(data_dir, 'analysis')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    analysis = sys.argv[1]
    if analysis == 'process':
        log_file = os.path.join(out_dir, 'Process_Analysis.log')
        process(data_dir, out_dir, log_file)
    elif analysis == 'quartiles':
        log_file = os.path.join(out_dir, 'Quartile_Analysis.log')
        quartiles(data_dir, out_dir, log_file)
    else:
        print("ERROR: 'analysis' should be 'process' or 'quartiles'")
