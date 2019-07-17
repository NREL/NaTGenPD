"""
Script to plot final fits for all CC units
"""
import numpy as np
import os
import plotting as mplt
import seaborn as sns
import sys

pkg_dir = '/home/mrossol/NaTGenPD'
sys.path.append(pkg_dir)
from NaTGenPD.analysis import ProcedureAnalysis


def get_hr_fit(group_fits, unit_id):
    """
    Get heat rate fit
    """
    unit_fit = group_fits.loc[group_fits['unit_id'] == unit_id]
    cols = ['load_min', 'load_2', 'load_3', 'load_4', 'load_max']
    load = unit_fit[cols].values
    hr = unit_fit[['heat_rate({})'.format(c) for c in cols]].values
    try:
        fit = np.dstack((load, hr))[0]
    except IndexError:
        fit = np.array([[0, 0], [0, 0]])

    return fit


def get_raw_cc(raw_group, unit_id):
    """
    Get Raw CC unit
    """
    pos = raw_group.df['cc_unit'] == unit_id
    cc_unit = raw_group.df.loc[pos]
    cc_unit = cc_unit.groupby('time')[['load', 'HTINPUT']].sum()
    cc_unit = cc_unit.reset_index()

    cc_unit['heat_rate'] = cc_unit['HTINPUT'] / cc_unit['load']
    return cc_unit[['load', 'heat_rate']].values


def lighten(color, perc):
    """
    lighten color by given perc
    """
    color = [min(c + (1 * perc), 1) for c in color]
    return color


def darken(color, perc):
    """
    darken color by given perc
    """
    color = [max(c - (1 * perc), 0) for c in color]
    return color


def plot_unit(unit_id, cc_filtered, cc_fits, out_dir):
    """
    Plot cc unit with fits
    """
    title = unit_id
    try:
        filtered_unit = cc_filtered[unit_id]
        pos = filtered_unit['cluster'] >= 0
        clusters = len(filtered_unit.loc[pos, 'cluster'].unique())
        colors = sns.color_palette('colorblind', clusters)
        cluster_units = []
        cluster_fits = []
        legend = []
        for c, cluster_df in filtered_unit.loc[pos].groupby('cluster'):
            cts = int(cluster_df.iloc[0]['cts'])
            legend.append('CTs = {}'.format(cts))
            cluster_units.append(cluster_df[['load', 'heat_rate']].values)
            colors.append(lighten(colors[c], 0.5))
            cluster_id = '{}-{}'.format(unit_id, c)
            cluster_fits.append(get_hr_fit(cc_fits, cluster_id))

        plt_data = cluster_units + cluster_fits
        linestyles = ('', ) * len(legend) + ('--',) * len(legend)
        f_path = os.path.join(out_dir, '{}.png'.format(unit_id))
        x = filtered_unit.loc[pos, 'load'].values
        x_lim = np.nanmax(x[x != np.inf]) * 1.1
        y = filtered_unit.loc[pos, 'heat_rate'].values
        y_lim = np.nanmax(y[y != np.inf]) * 1.1
        mplt.line_plot(*plt_data, despine=True, title=title,
                       linestyles=linestyles, markers=('o', ),
                       colors=colors,
                       xlabel='Load (MWh)', ylabel='Heat Rate (mmBTU/MWh)',
                       xlim=(0, x_lim), ylim=(0, y_lim),
                       legend=legend, legend_loc=0,
                       filename=f_path, showplot=False
                       )
    except Exception as ex:
        print("{} failed due to {}".format(unit_id, ex))


if __name__ == '__main__':
    data_dir = '/scratch/mrossol/CEMS'
    out_dir = os.path.join(data_dir, 'analysis', 'Figures', 'All_CCs')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fits_dir = os.path.join(data_dir, 'Final_Fits')
    raw_paths = [os.path.join(data_dir, '{y}/SMOKE_{y}.h5'.format(y=y))
                 for y in (2016, 2017)]
    clean_path = os.path.join(data_dir, 'SMOKE_Clean_2016-2017.h5')
    filter_path = os.path.join(data_dir, 'SMOKE_Filtered_2016-2017.h5')
    analysis = ProcedureAnalysis(fits_dir, raw_paths, clean_path, filter_path)

    group_type = 'CC (NG)'
    cc_raw = analysis._get_raw(group_type)
    cc_filtered = analysis._get_filtered(group_type)
    cc_fits = analysis._fits[group_type]
    cc_fits = cc_fits.loc[~cc_fits['a0'].isnull()]

    cc_ids = (cc_fits['unit_id'].str.split('-').str[0]).unique()
    for unit_id in cc_ids:
        plot_unit(unit_id, cc_filtered, cc_fits, out_dir)
        print('{} plotted'.format(unit_id))
