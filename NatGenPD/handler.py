# -*- coding: utf-8 -*-
"""
Wrapper on .h5 to handle CEMS data

@author: mrossol
"""
import h5py


class CEMS:
    """
    Class to handle CEMS data .h5 files
    """
    def __init__(self, h5_path, mode='r'):
        self._h5 = h5py.File(h5_path, mode=mode)
