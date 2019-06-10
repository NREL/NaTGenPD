"""
HPC Script to parse, clean, filter, and fit CEMS data
"""
import os
import sys
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)
import NaTGenPD as npd


def filter(data_dir, log_file):
    """
    Parse, clean, filter, and fit all units for the given years

    Parameters
    ----------
    data_dir : str
        Directory containing SMOKE data
    log_file : str
        Path to file to use for logging
    years : tuple
        Years to parse, clean, combine, filter, and fit
    """
    logger = npd.setup_logger('NaTGenPD', log_level='INFO',
                              log_file=log_file)
    comb_file = os.path.join(data_dir, 'SMOKE_Clean_2016-2017.h5')

    logger.info('Filter Clean SMOKE Data')
    filtered_file = os.path.join(data_dir, 'SMOKE_Filtered_2016-2017.h5')
    npd.Filter.run(comb_file, filtered_file, years=2)

    fit_dir = os.path.join(data_dir, 'CEMS_Fits')
    if not os.path.exists(fit_dir):
        os.makedirs(fit_dir)

    npd.PolyFit.run(filtered_file, fit_dir, order=4)


if __name__ == '__main__':
    data_dir = '/scratch/mrossol/CEMS'
    log_file = os.path.join(data_dir, 'Filter_CEMS_2016-2017.log')
    filter(data_dir, log_file)
