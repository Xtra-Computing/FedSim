import os
import abc
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import deprecation

from .SimModel import SimModel


class ThresholdSimModel(SimModel):
    def __init__(self, num_common_features, sim_threshold=0.0, **kwargs):
        super().__init__(num_common_features, **kwargs)
        self.sim_threshold = sim_threshold

    def match(self, data1, data2, labels, idx=None, preserve_key=False, sim_threshold=0.0, grid_min=-3., grid_max=3.01,
              grid_width=0.2, knn_k=3, tree_leaf_size=40, radius=0.1) -> tuple:
        [matched_data1, matched_data2], ordered_labels, data_indices = \
            super().match(data1, data2, labels, idx=idx, preserve_key=preserve_key, sim_threshold=self.sim_threshold)
        # remove similarity score from data
        return (matched_data1[:, 1:], matched_data2[:, 1:]), ordered_labels, data_indices