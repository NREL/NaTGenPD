# -*- coding: utf-8 -*-
"""
Wrapper on .h5 to handle CEMS data
@author: mrossol
"""
import h5py
import logging
import numpy as np
import os
import pandas as pd

logger = logging.getLogger(__name__)


class CEMSGroup:
    """
    Class to handle DataFrame of CEMS data for a single fuel/technology type
    """
    def __init__(self, group_data):
        self._df = self._parse_group(group_data)
        self._unit_dfs = self._df.groupby('unit_id')
        self._i = 0

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self.group_type)
        return msg

    def __getitem__(self, unit_id):
        if unit_id not in self.units:
            raise KeyError('{} is not a valid unit id'
                           .format(unit_id))

        unit_df = self._unit_dfs.get_group(unit_id)
        unit_df = unit_df.loc[unit_df['load'] > 0]

        return unit_df

    def __len__(self):
        return len(self.units)

    def __iter__(self):
        return self

    def __next__(self):
        if self._i < len(self):
            unit = self.units[self._i]
            unit_df = self[unit]
            self._i += 1
            return unit, unit_df
        else:
            raise StopIteration

    @property
    def df(self):
        """
        DataFrame of group data

        Returns
        -------
        df : pd.DataFrame
            DataFrame of group data
        """
        return self._df

    @property
    def units(self):
        """
        Units in group

        Returns
        -------
        units : ndarray
            List of units present in the group
        """
        return self.df['unit_id'].unique()

    @property
    def group_type(self):
        """
        Group type for data

        Returns
        -------
        group : str
            Fuel/technology type for units
        """
        return self.df['group_type'].unique()[0]

    @staticmethod
    def _parse_rec_arrary(arr):
        """
        Convert records array to DataFrame, decoding byte

        Parameters
        ----------
        arr : np.rec.array
            Records array of group data

        Returns
        -------
        df : pd.DataFrame
            DataFrame of group data
        """
        df = pd.DataFrame()
        for col in arr.dtype.names:
            data = arr[col]
            if np.issubdtype(data.dtype, np.bytes_):
                data = np.char.decode(data, encoding='utf-8')

            df[col] = data

        return df

    @staticmethod
    def _parse_group(group_data):
        """
        Parse group data

        Parameters
        ----------
        group_data : pd.DataFrame | np.rec.array | CEMSGroup
            Records array of group data

        Returns
        -------
        group_data : pd.DataFrame
            DataFrame of group data
        """
        if isinstance(group_data, np.ndarray):
            group_data = CEMSGroup._parse_rec_arrary(group_data)
        elif isinstance(group_data, CEMSGroup):
            group_data = group_data.df
        elif isinstance(group_data, pd.DataFrame):
            group_data = group_data
        else:
            raise ValueError('Cannot parse DataFrame from group_data of type'
                             ' {}'.format(type(group_data)))

        return group_data


class CEMS:
    """
    Class to handle CEMS data .h5 files
    """
    def __init__(self, h5_path, mode='r'):
        """
        Parameters
        ----------
        h5_path : str
            Path to CEMS .h5 file
            NOTE: CEMS class cannot handle Raw SMOKE .h5 produced by ParseSmoke
        mode : str
            Mode with which to open h5py File instance
        """
        self._h5_file = os.path.basename(h5_path)
        self._h5 = h5py.File(h5_path, mode=mode)
        self._mode = mode

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self._h5_file)
        return msg

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

        if type is not None:
            raise

    def __getitem__(self, dset):
        if dset in self.dsets:
            group_df = self._h5[dset][...]
        else:
            raise KeyError('{} is not a valid group_type'
                           .format(dset))

        return CEMSGroup(group_df)

    def __setitem__(self, dset, arr):
        if self.writable:
            if isinstance(arr, CEMSGroup):
                arr = CEMSGroup.df

            if isinstance(arr, pd.DataFrame):
                arr = self.to_records_array(arr)

            self.update_dset(dset, arr)

    @property
    def writable(self):
        """
        Check to see if h5py.File instance is writable
        Returns
        -------
        bool
            Flag if mode is writable
        """
        mode = ['a', 'w', 'w-', 'x']
        if self._mode not in mode:
            msg = 'mode must be writable: {}'.format(mode)
            raise RuntimeError(msg)

        return True

    @property
    def dsets(self):
        """
        Datasets available in .h5 file

        Returns
        -------
        list
            List of group_types stored as individual DataSets
        """
        return list(self._h5)

    @staticmethod
    def get_dtype(col):
        """
        Get column dtype for converstion to records array

        Parameters
        ----------
        col : pandas.Series
            Column from pandas DataFrame

        Returns
        -------
        str
            converted dtype for column
            -  float = float32
            -  int = int16 or int32 depending on data range
            -  object/str = U* max length of strings in col
        """
        dtype = col.dtype
        if np.issubdtype(dtype, np.floating):
            out = 'float32'
        elif np.issubdtype(dtype, np.integer):
            if col.max() < 32767:
                out = 'int16'
            else:
                out = 'int32'
        elif np.issubdtype(dtype, np.object_):
            col = col.astype(str)
            size = int(col.str.len().max())
            out = 'S{:}'.format(size)
        elif np.issubdtype(dtype, np.datetime64):
            col = col.astype(str)
            size = int(col.str.len().max())
            out = 'S{:}'.format(size)
        else:
            out = dtype

        return out

    @staticmethod
    def to_records_array(df):
        """
        Convert pandas DataFrame to numpy Records Array

        Parameters
        ----------
        df : pandas.DataFrame
            Pandas DataFrame to be converted

        Returns
        -------
        numpy.rec.array
            Records array of input df
        """
        meta_arrays = []
        dtypes = []
        for c_name, c_data in df.iteritems():
            dtype = CEMS.get_dtype(c_data)
            if np.issubdtype(dtype, np.bytes_):
                c_data = c_data.astype(str).fillna('None')
                c_data = c_data.str.encode('utf-8').values
            else:
                c_data = c_data.fillna(0)
                c_data = c_data.values

            arr = np.array(c_data, dtype=dtype)
            meta_arrays.append(arr)
            dtypes.append((c_name, dtype))

        return np.core.records.fromarrays(meta_arrays, dtype=dtypes)

    def update_dset(self, dset, arr):
        """
        Save dset to disk, if needed create dataset

        Parameters
        ----------

        """
        if dset in self.dsets:
            self._h5[dset][...] = arr
        else:
            self._h5.create_dataset(dset, shape=arr.shape, dtype=arr.dtype,
                                    data=arr)

    def close(self):
        """
        Close h5 instance
        """
        self._h5.close()

    @classmethod
    def combine_years(cls, comb_file, year_files):
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
        years = [os.path.basename(f) for f in year_files]
        logger.info('Combining data from {} into {}'
                    .format(years, comb_file))
        with cls(comb_file, mode='w') as f_comb:
            dsets = []
            for file in year_files:
                with CEMS(file, mode='r') as f:
                    dsets.extend(f.dsets)

            dsets = list(set(dsets))
            for ds in dsets:
                comb_arr = []
                for file in year_files:
                    with CEMS(file, mode='r') as f:
                        if ds in f.dsets:
                            comb_arr.append(f._h5[ds][...])
                            logger.debug('- {} extracted from {}'
                                         .format(ds, os.path.basename(file)))

                comb_arr = np.hstack(comb_arr)
                f_comb.update_dset(ds, comb_arr)
                logger.info('{} combined'.format(ds))
