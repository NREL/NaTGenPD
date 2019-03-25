# -*- coding: utf-8 -*-
"""
Wrapper on .h5 to handle CEMS data

@author: mrossol
"""
import h5py
import os
import pandas as pd


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

    def close(self):
        """
        Close h5 instance
        """
        self._h5.close()

    def __getitem__(self, key):
        if key in self.dsets:
            group_df = self._h5[key][...]
        else:
            raise KeyError('{} is not a valid group_type'
                           .format(key))

        return pd.DataFrame(group_df)

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
