import os
import abc
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import deprecation

from .OnePartyModel import BaseModel


class TwoPartyBaseModel(abc.ABC, BaseModel):
    def __init__(self, num_common_features, drop_key=True, grid_min=-3., grid_max=3.01, grid_width=0.2,
                 knn_k=3, tree_leaf_size=40, kd_tree_radius=0.1,
                 dataset_type='syn', **kwargs):

        super().__init__(**kwargs)
        assert dataset_type in ['syn', 'real']
        self.dataset_type = dataset_type
        self.drop_key = drop_key
        self.num_common_features = num_common_features
        self.tree_radius = kd_tree_radius
        self.tree_leaf_size = tree_leaf_size
        self.knn_k = knn_k
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.grid_width = grid_width
        self.sim_scaler = None

    @abc.abstractmethod
    def match(self, data1, data2, labels, idx=None, preserve_key=False, sim_threshold=0.0,
              grid_min=-3., grid_max=3.01, grid_width=0.2, knn_k=3, tree_leaf_size=40, radius=0.1) -> tuple:
        """
        Match the data of two parties, return the matched data
        :param radius:
        :param knn_k:
        :param tree_leaf_size:
        :param idx: Index of data1, only for evaluation. It should not be involved in linkage.
        :param sim_threshold: threshold of similarity score, everything below the threshold will be removed
        :param data1: data in party 1
        :param data2: data in party 2
        :param labels: labels (in party 1)
        :param preserve_key: whether to preserve common features in the output
        :return: [matched_data1, matched_data2], matched_labels
                 Each line refers to one sample
        """
        raise NotImplementedError

    # def train_combine(self, data1, data2, labels, data_cache_path=None):
    #     train_X, val_X, test_X, train_y, val_y, test_y, train_idx, val_idx, test_idx = \
    #         self.prepare_train_combine(data1, data2, labels, data_cache_path)
    #
    #     return self._train(train_X, val_X, test_X, train_y, val_y, test_y,
    #                        train_idx[:, 0], val_idx[:, 0], test_idx[:, 0])



