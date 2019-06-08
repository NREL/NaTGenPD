"""
HPC Script to parse, clean, filter, and fit CEMS data
"""
import os
import sys
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PKG_DIR)
import NaTGenPD as npd


def run(data_dir, log_file, years=(2016, 2017)):
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
    for year in years:
        logger.info('Parsing {} SMOKE Data'.format(year))
        dir_path = os.path.join(data_dir, str(year))
        npd.ParseSmoke.performance_vars(dir_path, year, save=True)

    unit_attrs_path = os.path.join(PKG_DIR, 'bin', 'emission_01-17-2017.csv')
    cc_map_path = os.path.join(PKG_DIR, 'bin', 'cems_cc_mapping.csv')

    for year in years:
        logger.info('Cleaning {} SMOKE Data'.format(year))
        smoke_path = os.path.join(data_dir, '{y}/SMOKE_{y}.h5'.format(y=year))
        print('Cleaning {}'.format(os.path.basename(smoke_path)))
        out_path = os.path.join(data_dir, 'SMOKE_Clean_{}.h5'.format(year))
        _ = npd.CleanSmoke.clean(smoke_path, unit_attrs_path=unit_attrs_path,
                                 cc_map=cc_map_path, out_file=out_path)

    logger.info('Combining Clean SMOKE Data')
    year_files = []
    for year in years:
        year_files.append(os.path.join(data_dir,
                                       'SMOKE_Clean_{}.h5'.format(year)))

    comb_file = os.path.join(data_dir, 'SMOKE_Clean_2016-2017.h5')
    npd.CEMS.combine_years(comb_file, year_files)

    logger.info('Filter Clean SMOKE Data')
    filtered_file = os.path.join(data_dir, 'SMOKE_Filtered_2016-2017.h5')
    npd.Filter.run(comb_file, filtered_file, years=2)

    fit_dir = os.path.join(data_dir, 'CEMS_Fits')
    if not os.path.exists(fit_dir):
        os.makedirs(fit_dir)

    npd.PolyFit.run(filtered_file, fit_dir, order=4)


if __name__ == '__main__':
    data_dir = '/scratch/mrossol/CEMS'
    log_file = os.path.join(data_dir, 'CEMS_2016-2017.log')
    run(data_dir, log_file)
