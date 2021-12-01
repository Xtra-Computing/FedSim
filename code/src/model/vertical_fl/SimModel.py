import os
import abc
import pickle
import warnings
import gc
from datetime import datetime
from queue import PriorityQueue
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KDTree, BallTree, NearestNeighbors
from scipy.stats import binned_statistic_dd
from joblib import Parallel, delayed

import torch
from torch.utils.data import Dataset

from tqdm import tqdm
import deprecation
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from captum.attr import IntegratedGradients
from lhbf import BloomFilter
from nltk import ngrams
import faiss
from phe import paillier

from .TwoPartyModel import TwoPartyBaseModel
from utils import scaled_edit_distance, DroppingPriorityQueue
from utils.privacy import SimNoiseScale, l2_distance_with_he


def remove_conflict(matched_indices, n_batches: int):
    """
    Remove conflict based on equitable coloring
    :param matched_indices:
    :param n_batches:
    :return:
    """
    # build conflict graph
    print("Adding edges")
    conflict_records = {}
    conflict_edges = []
    for a, b in tqdm(matched_indices):
        a = int(a)
        b = int(b)
        if b in conflict_records:
            # add conflicts to graph
            conflict_nodes = conflict_records[b]
            conflict_edges += [(a, v) for v in conflict_nodes]

            # record new node
            conflict_records[b].add(a)
        else:
            # no conflict
            conflict_records[b] = {a}
    print("Constructing conflict graph")
    start_time = datetime.now()
    G = nx.Graph()
    G.add_edges_from(conflict_edges)
    time_cost_sec = (datetime.now() - start_time).seconds
    print("Done constructing, cost {} seconds".format(time_cost_sec))

    max_degree = max(G.degree, key=lambda x: x[1])[1]
    if max_degree > n_batches:
        warnings.warn("Degree {} is higher than #colors {}, coloring might cost exponential time"
                      .format(max_degree, n_batches))

    # equitable color the graph
    print("Coloring")
    start_time = datetime.now()
    a_colors = nx.equitable_color(G, n_batches)
    time_cost_sec = (datetime.now() - start_time).seconds
    print("Done coloring, cost {} seconds".format(time_cost_sec))

    sorted_a_colors = list(sorted(a_colors.item(), key=lambda x: x[0]))
    return sorted_a_colors


# def remove_conflict(matched_indices: np.ndarray, n_batches: int):
#     # group by idx of B
#     sorted_indices = np.sort(matched_indices, axis=1)
#
#     batched_indices = [[] for _ in n_batches]
#     cur_a_count = 0
#     cur_a = -1
#     cur_pos = 0
#     for idx1, idx2 in sorted_indices:
#         batched_indices[cur_pos].append([idx1, idx2])
#
#         if int(idx1) == int(cur_a):
#             cur_a_count += 1
#             if cur_a_count > n_batches:
#                 assert False, "The degree of {} is larger than batch size {}".format(idx2, n_batches)
#         else:
#             cur_a = idx1
#             cur_a_count = 1
#
#         cur_pos = (cur_pos + 1) % n_batches
#
#     return batched_indices


class SimDataset(Dataset):
    @torch.no_grad()  # disable auto_grad in __init__
    def __init__(self, data1, data2, labels, data_idx, sim_dim=1):
        self.sim_dim = sim_dim

        # data1[:, 0] and data2[:, 0] are both sim_scores
        assert data1.shape[0] == data2.shape[0] == data_idx.shape[0]
        # remove similarity scores in data1 (at column 0)
        data1_labels = np.concatenate([data1[:, sim_dim:], labels.reshape(-1, 1)], axis=1)

        print("Grouping data")
        grouped_data1 = {}
        grouped_data2 = {}
        for i in tqdm(range(data_idx.shape[0])):
            idx1, idx2 = data_idx[i]
            new_data2 = np.concatenate([idx2.reshape(1, 1), data2[i].reshape(1, -1)], axis=1)
            if idx1 in grouped_data2:
                grouped_data2[idx1].append(new_data2)
            else:
                grouped_data2[idx1] = [new_data2]
            np.random.shuffle(grouped_data2[idx1])  # shuffle to avoid information leakage of the order
            grouped_data1[idx1] = data1_labels[i]
        for k, v in tqdm(grouped_data2.items()):
            grouped_data2[k] = np.vstack(v)
        print("Done")

        print("Checking if B is sorted by similarity: ", end="")
        is_sorted = True
        for k, v in grouped_data2.items():
            is_sorted = is_sorted and np.all(np.diff(v[:, 1].flatten()) < 0)
        print(is_sorted)

        group1_data_idx = np.array(list(grouped_data1.keys()))
        group1_data1_labels = np.array(list(grouped_data1.values()))
        group2_data_idx = np.array(list(grouped_data2.keys()))
        group2_data2 = np.array(list(grouped_data2.values()), dtype='object')

        # print("Equitable coloring to remove conflict")
        # data1_colors = remove_conflict(data_idx, n_batches=np.ceil(data1.shape[0] / batch_sizes).astype('int'))
        # data1_order = np.argsort(data1_colors, axis=1)

        print("Sorting data")
        group1_order = group1_data_idx.argsort()
        group2_order = group2_data_idx.argsort()

        group1_data_idx = group1_data_idx[group1_order]
        group1_data1_labels = group1_data1_labels[group1_order]
        group2_data_idx = group2_data_idx[group2_order]
        group2_data2 = group2_data2[group2_order]
        assert (group1_data_idx == group2_data_idx).all()
        print("Done")

        self.data1_idx: np.ndarray = group1_data_idx
        data1: np.ndarray = group1_data1_labels[:, :-1]
        self.labels: torch.Tensor = torch.from_numpy(group1_data1_labels[:, -1])
        data2: list = group2_data2

        print("Retrieve data")
        data_list = []
        weight_list = []
        data_idx_list = []
        data2_idx_list = []
        self.data_idx_split_points = [0]
        for i in tqdm(range(self.data1_idx.shape[0])):
            d2 = torch.from_numpy(data2[i].astype(np.float)[:, 1:])  # remove index
            d2_idx = data2[i].astype(np.float)[:, 0].reshape(-1, 1)
            d1 = torch.from_numpy(np.repeat(data1[i].reshape(1, -1), d2.shape[0], axis=0))
            d = torch.cat([d2[:, :sim_dim], d1, d2[:, sim_dim:]], dim=1)  # move similarity to index 0
            data_list.append(d)

            weight = torch.ones(d2.shape[0]) / d2.shape[0]
            weight_list.append(weight)

            d1_idx = np.repeat(self.data1_idx[i].item(), d2.shape[0], axis=0)
            # idx = torch.from_numpy(np.concatenate([d1_idx.reshape(-1, 1), d2_idx], axis=1))
            data2_idx_list.append(torch.from_numpy(d2_idx))
            data_idx_list.append(torch.from_numpy(d1_idx))
            self.data_idx_split_points.append(self.data_idx_split_points[-1] + d1_idx.shape[0])
        print("Done")

        self.data = torch.cat(data_list, dim=0)  # sim_scores; data1; data2
        self.weights = torch.cat(weight_list, dim=0)
        self.data_idx = torch.cat(data_idx_list, dim=0)
        self.data2_idx = torch.cat(data2_idx_list, dim=0)

    def __len__(self):
        return self.data1_idx.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        start = self.data_idx_split_points[idx]
        end = self.data_idx_split_points[idx + 1]

        return self.data[start:end], self.labels[idx], self.weights[start:end], \
               self.data_idx[start:end], self.data1_idx[idx]

    def add_noise_to_sim_(self, noise_scale=0.0):
        print("Adding noise of scale {} to sim_scores".format(noise_scale))
        sim_noise = np.random.normal(0, scale=noise_scale, size=(self.data.shape[0], self.sim_dim))
        self.data[:, :self.sim_dim] += sim_noise.astype('float64')

    @property
    @torch.no_grad()
    def top1_dataset(self):
        print("Generating top1 dataset")
        X = torch.empty([self.data1_idx.shape[0], self.data.shape[1] - self.sim_dim], dtype=self.data.dtype)
        y = torch.empty(self.labels.shape[0], dtype=self.labels.dtype)
        idx = np.empty(self.data1_idx.shape[0], dtype=self.data1_idx.dtype)
        for i in tqdm(range(self.data1_idx.shape[0])):
            start = self.data_idx_split_points[i]
            end = self.data_idx_split_points[i + 1]

            max_i = torch.argmax(self.data[start:end][:, 0])
            X_i = self.data[start:end][max_i, self.sim_dim:]
            y_i = self.labels[i]
            idx_i = self.data1_idx[i]
            X[i] = X_i
            y[i] = y_i
            idx[i] = idx_i

        return X, y, idx

    @torch.no_grad()
    def filter_to_topk_dataset_(self, k):
        assert 0 < k <= self.data.shape[0] // self.data1_idx.shape[0]
        print("Generating top {} dataset".format(k))
        _data = torch.empty([self.data1_idx.shape[0] * k, self.data.shape[1]], dtype=self.data.dtype)
        # _labels = torch.empty(self.labels.shape[0], dtype=self.labels.dtype)
        _weights = torch.empty(self.data1_idx.shape[0] * k, dtype=self.weights.dtype)
        _data_idx = torch.empty(self.data1_idx.shape[0] * k, dtype=self.data_idx.dtype)
        # _data1_idx = np.empty(self.data1_idx.shape[0], dtype=self.data1_idx.dtype)
        _data_idx_split_points = [0]
        for i in tqdm(range(self.data1_idx.shape[0])):
            start = self.data_idx_split_points[i]
            end = self.data_idx_split_points[i + 1]
            new_start = _data_idx_split_points[-1]
            new_end = new_start + k

            _, top_indices = torch.topk(self.data[start:end][:, 0], k, sorted=False)
            _data[new_start:new_end] = self.data[start:end][top_indices]
            _weights[new_start:new_end] = self.weights[start:end][top_indices]
            _data_idx[new_start:new_end] = self.data_idx[start:end][top_indices]

            _data_idx_split_points.append(new_end)

        self.data = _data
        self.weights = _weights
        self.data_idx = _data_idx
        self.data_idx_split_points = _data_idx_split_points


class SimModel(TwoPartyBaseModel):
    def __init__(self, num_common_features, n_clusters=100, center_threshold=0.5,
                 blocking_method='grid', feature_wise_sim=False, psig_p=7,
                 # Split Learning
                 local_hidden_sizes=None, agg_hidden_sizes=None, cut_dims=None,
                 edit_distance_threshold=1, n_hash_func=10, collision_rate=0.05,
                 qgram_q=2, link_delta=0.1, n_hash_lsh=20, link_threshold_t=0.1,
                 link_epsilon=0.1, sim_leak_p=0.0, link_n_jobs=1,

                 filter_top_k=None,

                 **kwargs):
        super().__init__(num_common_features, **kwargs)

        self.filter_top_k = filter_top_k
        self.link_n_jobs = link_n_jobs
        self.psig_p = psig_p
        self.sim_leak_p = sim_leak_p
        self.link_threshold_t = link_threshold_t
        self.link_epsilon = link_epsilon
        self.n_hash_lsh = n_hash_lsh
        self.edit_distance_threshold = edit_distance_threshold
        self.n_hash_func = n_hash_func
        self.collision_rate = collision_rate
        self.qgram_q = qgram_q
        self.link_delta = link_delta
        self.feature_wise_sim = feature_wise_sim
        self.blocking_method = blocking_method
        self.center_threshold = center_threshold
        self.n_clusters = n_clusters

        if local_hidden_sizes is None:
            self.local_hidden_sizes = [[10], [10]]
        else:
            self.local_hidden_sizes = local_hidden_sizes

        if cut_dims is None:
            self.cut_dims = [100, 10]
        else:
            self.cut_dims = cut_dims

        if agg_hidden_sizes is None:
            self.agg_hidden_sizes = [10]
        else:
            self.agg_hidden_sizes = agg_hidden_sizes

        if "priv" not in blocking_method:
            assert np.isclose(self.sim_leak_p, 1.0), "Noise will be added while no privacy is required"

        self.scale_analysis = None

    def merge_pred(self, pred_all: list):
        sort_pred_all = list(sorted(pred_all, key=lambda t: t[0], reverse=True))
        return sort_pred_all[0][0]

    @staticmethod
    def _collect(array):
        """
        Obtained from
        https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
        :param array:
        :return: <Values of unique elements>, <
        """
        idx_sort = np.argsort(array)
        sorted_array = array[idx_sort]

        # returns the unique values, the index of the first occurrence of a value, and the count for each element
        vals, idx_start, count = np.unique(sorted_array, return_counts=True, return_index=True)

        # splits the indices into separate arrays
        res = np.split(idx_sort, idx_start[1:])

        return vals, res

    @deprecation.deprecated()
    def __cal_sim_score_kmeans(self, key1, key2, seed=0):
        """
        Deprecated
        :return: numpy array with size (n, 3), table of similarity scores
                 ['index_data1': int, 'index_data2': int, 'sim_score': float]
        """

        # clustering
        print("Clustering")
        kmeans1 = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=seed).fit(key1)
        kmeans2 = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=seed).fit(key2)

        # filter close cluster centers
        print("Filter close cluster centers")
        close_clusters = []
        for i, center1 in enumerate(kmeans1.cluster_centers_):
            for j, center2 in enumerate(kmeans2.cluster_centers_):
                if np.linalg.norm(center1 - center2) < self.center_threshold:
                    close_clusters.append((i, j))
        print("Got {} close clusters after filtering".format(len(close_clusters)))

        labels1, indices1 = self._collect(kmeans1.labels_)
        labels2, indices2 = self._collect(kmeans2.labels_)

        # compare within the block
        print("Compare within each cluster")
        sim_scores = []
        for (label1, label2) in close_clusters:
            idx1 = indices1[np.argwhere(labels1 == label1).item()]
            idx2 = indices2[np.argwhere(labels2 == label2).item()]
            for i in idx1:
                for j in idx2:
                    if self.feature_wise_sim:
                        score = -(key1[i] - key2[j]) ** 2
                    else:
                        score = -np.linalg.norm(key1[i] - key2[j])  # reverse distance
                    sim_scores.append(np.concatenate([np.array([i, j]), np.array(score)], axis=0))
        print("Done calculating similarity scores")

        # scale similarity scores to [0, 1]
        sim_scores = np.stack(sim_scores)
        if self.sim_scaler is not None:
            sim_scores[:, 2:] = self.sim_scaler.transform(sim_scores[:, 2:])
        else:
            self.sim_scaler = StandardScaler()
            sim_scores[:, 2:] = self.sim_scaler.fit_transform(sim_scores[:, 2:])
        print("Done scaling")

        return np.array(sim_scores)

    @staticmethod
    def _array2base(array, base):
        res = 0
        if hasattr(base, "__getitem__"):
            for i, num in enumerate(array[::-1]):
                res += num * np.prod(base[::-1][:i]).astype('int')
        else:
            for i, num in enumerate(array[::-1]):
                res += num * base ** i

        return res

    def cal_sim_score_grid(self, key1, key2, grid_min=-3., grid_max=3.01, grid_width=0.2):
        """
        :param key1: Common features in party 1
        :param key2: Common features in party 2
        :param grid_min: min value of grid
        :param grid_max: min value of grid
        :param grid_width: width of grid
        :return: sim_scores: Nx3 np.ndarray (idx_key1, idx_key2, sim_score)
        """
        print("Quantization")
        bins = np.arange(grid_min, grid_max, grid_width)
        quantized_key1 = np.digitize(key1, bins)
        quantized_key2 = np.digitize(key2, bins)
        quantized_key1 = np.array([self._array2base(k, bins.shape[0] + 1) for k in quantized_key1])
        quantized_key2 = np.array([self._array2base(k, bins.shape[0] + 1) for k in quantized_key2])

        print("Collect unique values")
        grid_ids1, indices1 = self._collect(quantized_key1)
        grid_ids2, indices2 = self._collect(quantized_key2)

        blocks = np.intersect1d(quantized_key1, quantized_key2)
        print("Finished quantization, got {} blocks".format(blocks.shape[0]))

        # compare within the block
        print("Compare within each block")
        sim_scores = []
        for quantized_id in tqdm(blocks):
            idx1 = indices1[np.argwhere(grid_ids1 == quantized_id).item()]
            idx2 = indices2[np.argwhere(grid_ids2 == quantized_id).item()]
            for i in idx1:
                for j in idx2:
                    if self.feature_wise_sim:
                        score = -(key1[i] - key2[j]) ** 2
                    else:
                        score = -np.linalg.norm(key1[i] - key2[j]).reshape(-1)  # reverse distance
                    sim_scores.append(np.concatenate([np.array([i, j]), np.array(score)], axis=0))
        print("Done calculating similarity scores")

        # scale similarity scores to [0, 1]
        sim_scores = np.stack(sim_scores)
        if self.sim_scaler is not None:
            sim_scores[:, 2:] = self.sim_scaler.transform(sim_scores[:, 2:])
        else:
            self.sim_scaler = StandardScaler()
            sim_scores[:, 2:] = self.sim_scaler.fit_transform(sim_scores[:, 2:])
        print("Done scaling")

        return sim_scores

    def cal_sim_score_knn_he(self, key1, key2, knn_k=3, grid_min=0, grid_max=1, grid_width=0.1,
                             block_eps=0.5, num_tolerate_comparisons=5000, fake_point=1000):
        """

        :param key1:
        :param key2:
        :param knn_k:
        :param grid_min:
        :param grid_max:
        :param grid_width:
        :param block_eps:
        :param num_tolerate_comparisons:
        :param fake_point: must be a number that is far from all existing records.
                          This is used to generate fake records.
        :return:
        """
        if not hasattr(grid_min, "__getitem__") or not hasattr(grid_width, "__getitem__") \
                or not hasattr(grid_max, "__getitem__"):
            grid_min = list(np.repeat(grid_min, key1.shape[1]))
            grid_width = list(np.repeat(grid_width, key1.shape[1]))
            grid_max = list(np.repeat(grid_max, key1.shape[1]))

        print("Quantize each dimensions of key1 and key2")
        bins = [np.arange(low, high, step) for low, high, step in zip(grid_min, grid_max, grid_width)]
        assert key1.shape[1] == key2.shape[1] == len(bins)
        quantized_key1 = []
        quantized_key2 = []
        for j in range(key2.shape[1]):
            quantized_key1_j = np.digitize(key1[:, j], bins[j])
            quantized_key2_j = np.digitize(key2[:, j], bins[j])
            quantized_key1.append(quantized_key1_j)
            quantized_key2.append(quantized_key2_j)
        quantized_key1 = np.vstack(quantized_key1).T
        quantized_key2 = np.vstack(quantized_key2).T
        bin_lens = [len(b) + 1 for b in bins]
        quantized_key1 = np.array([self._array2base(k, bin_lens) for k in quantized_key1])
        quantized_key2 = np.array([self._array2base(k, bin_lens) for k in quantized_key2])

        num_blocks = np.prod(bin_lens)
        print("Done. Got {} blocks".format(num_blocks))

        print("Collect indices of samples by grid")
        grid_ids1, indices1 = self._collect(quantized_key1)
        grid_ids2, indices2 = self._collect(quantized_key2)
        print("Done. Got {} valid grids for key1 and {} valid grids for key2"
              .format(grid_ids1.size, grid_ids2.size))

        print("Calculating blocking parameters")
        b = 2 / block_eps
        expected_samples_per_block = int(key2.shape[0] / num_blocks)
        mu = - b * np.log((2 * expected_samples_per_block) / (expected_samples_per_block + num_tolerate_comparisons))

        print("Noise mean {}, noise scale {}".format(mu, b))
        block_sizes_noises = np.round(np.random.laplace(mu, b, num_blocks))
        print("Noises of {} blocks: {}".format(num_blocks, block_sizes_noises))

        print("Adding noise to each block in party B")
        indices = indices2  # set party to add noise
        grid_ids = grid_ids2
        for block_id in tqdm(range(num_blocks)):
            noise = int(block_sizes_noises[block_id])
            if noise < 0:
                if block_id in grid_ids:
                    i = np.argwhere(grid_ids == block_id).item()
                    if abs(noise) <= len(indices[i]):
                        np.random.choice(indices[i], len(indices[i]) + noise, replace=True)
                    else:
                        indices[i] = np.array([])
            else:  # noise >= 0
                if block_id in grid_ids:
                    i = np.argwhere(grid_ids == block_id).item()
                    indices[i] = np.concatenate([indices[i], np.repeat(-1, noise)])  # -1 means fake sample
                else:
                    grid_ids = np.insert(grid_ids, grid_ids.size, block_id)
                    indices.append(np.repeat(-1, noise))
        print("Done. Got {} valid grids for key1 and {} valid grids for key2"
              .format(grid_ids1.size, grid_ids2.size))

        common_blocks = np.intersect1d(grid_ids1, grid_ids2)
        print("Got {} common blocks to compare".format(common_blocks.shape[0]))

        print("Generating keys for paillier")
        public_key, private_key = paillier.generate_paillier_keypair(n_length=512)

        # print("Calculating encrypted b")
        # enc_start = datetime.now()
        # encrypted_b = Parallel(n_jobs=self.link_n_jobs)(delayed(public_key.encrypt)(key2[i, j])
        #                                                 for i in range(key2.shape[0]) for j in range(key2.shape[1]))
        # encrypted_b = np.array(encrypted_b).reshape(key2.shape)
        # print("Done, cost {} seconds".format((datetime.now() - enc_start).seconds))
        # print("Calculating encrypted b square")
        # enc_start = datetime.now()
        # encrypted_b_square = Parallel(n_jobs=self.link_n_jobs)(delayed(public_key.encrypt)(key2[i, j] * key2[i, j])
        #                                                        for i in range(key2.shape[0]) for j in
        #                                                        range(key2.shape[1]))
        # encrypted_b_square = np.array(encrypted_b_square).reshape(key2.shape)
        # print("Done, cost {} seconds".format((datetime.now() - enc_start).seconds))

        if not hasattr(fake_point, "__getitem__"):
            fake_point_center = np.repeat(fake_point, key2.shape[1])
        else:
            fake_point_center = np.array(fake_point)
        print("Fake point {}".format(fake_point_center))

        # num_fake_points = np.sum((block_sizes_noises > 0) * block_sizes_noises)
        # fake_points = np.repeat(fake_point_center, num_fake_points) + \
        #               np.random.normal(0, 1, [num_fake_points, fake_point_center.shape[0]])
        # encrypted_fake_point = Parallel(n_jobs=self.link_n_jobs)(delayed(public_key.encrypt)(fake_points[i][j])
        #                                                          for i in fake_points.shap[0]
        #                                                          for j in fake_points.shape[1])
        # encrypted_fake_point_square = Parallel(n_jobs=self.link_n_jobs)(
        #     delayed(public_key.encrypt)(fake_points[i][j] * fake_points[i][j])
        #     for i in fake_points.shap[0] for j in fake_points.shape[1])

        print("Estimate number of comparisons")
        num_comparisons = 0
        for quantized_id in tqdm(common_blocks):
            idx1 = indices1[np.argwhere(grid_ids1 == quantized_id).item()]
            idx2 = indices2[np.argwhere(grid_ids2 == quantized_id).item()]
            num_comparisons += len(idx1) * len(idx2)
        print("Require {} comparisons".format(num_comparisons))

        print("Compare within each block")
        assert self.feature_wise_sim is False
        sim_scores = []
        for quantized_id in tqdm(common_blocks):
            idx1 = indices1[np.argwhere(grid_ids1 == quantized_id).item()]
            idx2 = indices2[np.argwhere(grid_ids2 == quantized_id).item()]

            def compare(i, j):
                if j != -1:
                    key1_i = key1[i]  # i can't be -1, there is no noise on party A
                    key2_j = key2[j]
                    score = -np.linalg.norm(key1_i - key2_j).reshape(-1)
                else:
                    key1_i = key1[i]
                    key2_j = fake_point_center
                    score = -np.linalg.norm(key1_i - key2_j).reshape(-1)
                return np.concatenate([np.array([i, j]), np.array(score)], axis=0)

            # def compare(i, j):
            #     key1_i = key1[i]  # can't be -1, there is no noise on party A
            #     key2_j = encrypted_b[j] if i != -1 else encrypted_fake_point[0]
            #     key2_j_square = encrypted_b_square[j] if j != -1 else encrypted_fake_point_square[0]
            #     score = l2_distance_with_he(key1_i, key2_j, key2_j_square)
            #     return np.concatenate([np.array([i, j]), np.array(score)], axis=0)

            n_jobs = 1 if len(idx1) * len(idx2) <= 100000 else self.link_n_jobs
            sim_scores_block = Parallel(n_jobs=n_jobs, batch_size=100000, pre_dispatch='all')(
                delayed(compare)(i, j) for i in idx1 for j in idx2)
            sim_scores += sim_scores_block

            # for i in idx1:
            #     for j in idx2:
            #         key1_i = key1[i] if i != -1 else fake_point1
            #         key2_j = key2[j] if j != -1 else fake_point2
            #         score = -np.linalg.norm(key1_i - key2_j).reshape(-1)
            #         sim_scores.append(np.concatenate([np.array([i, j]), np.array(score)], axis=0))
        print("Done calculating similarity scores")

        print("Filter top {} neighbors".format(knn_k))
        sim_scores_knn = dict(zip(range(key1.shape[0]), [DroppingPriorityQueue(maxsize=knn_k, reverse=True)
                                                         for _ in range(key1.shape[0])]))
        for k1, k2, score in tqdm(sim_scores):
            sim_scores_knn[k1].put((score, k2))

        print("Write to matrix")
        sim_scores_matrix = np.empty([key1.shape[0] * knn_k, 3])
        for i, (k1, k2_queue) in enumerate(sim_scores_knn.items()):
            sim_scores_matrix[i: i + knn_k, 0] = np.repeat(k1, knn_k)
            k2_matrix_w_score = np.array(
                [list(k2_queue.get()) if len(k2_queue) > 0 else [-1, -abs(fake_point_center[0])]
                 for _ in range(knn_k)]).reshape(-1, 2)[:, ::-1]
            sim_scores_matrix[i: i + knn_k, 1:] = k2_matrix_w_score

        # scale similarity scores to [0, 1]
        sim_scores = sim_scores_matrix
        if self.sim_scaler is not None:
            sim_scores[:, 2:] = self.sim_scaler.transform(sim_scores[:, 2:])
        else:
            print("Saving raw sim scores")
            np.save('cache/sim_scores_he_test.npy', sim_scores)
            self.sim_scaler = StandardScaler()
            sim_scores[:, 2:] = self.sim_scaler.fit_transform(sim_scores[:, 2:])
        print("Done scaling")

        return sim_scores

    def cal_sim_score_knn(self, key1, key2, knn_k=3, tree_leaf_size=40):
        """
        :param tree_leaf_size: leaf size to build kd-tree
        :param knn_k: number of nearest neighbors to be calculated
        :param key1: Common features in party 1
        :param key2: Common features in party 2
        :return: sim_scores: Nx3 np.ndarray (idx_key1, idx_key2, sim_score)
        """

        nbrs = NearestNeighbors(n_neighbors=knn_k, algorithm='kd_tree', leaf_size=tree_leaf_size,
                                n_jobs=self.link_n_jobs)
        print("Constructing tree")
        nbrs.fit(key2)

        print("Query tree")
        # sort_results is set to True by default since it will be randomly shuffled in dataset construction
        dists, idx2 = nbrs.kneighbors(key1, return_distance=True)

        repeat_times = [x.shape[0] for x in idx2]
        idx2 = np.concatenate(idx2[np.array(repeat_times) > 0])

        print("Calculate sim_scores")
        idx1 = np.repeat(np.arange(key1.shape[0]), knn_k)
        if self.feature_wise_sim:
            sims = -(key1[idx1] - key2[idx2]) ** 2
            sim_scores = np.concatenate([idx1.reshape(-1, 1), idx2.reshape(-1, 1), sims], axis=1)
        else:
            sim_scores = np.vstack([idx1, idx2.flatten(), -dists.flatten()]).T

        if self.sim_scaler is not None:
            # use train scaler
            sim_scores[:, 2:] = self.sim_scaler.transform(sim_scores[:, 2:])
        else:
            print("Saving raw sim scores")
            np.save('cache/sim_scores_no_priv_test.npy', sim_scores)
            # generate train scaler
            self.sim_scaler = StandardScaler()
            sim_scores[:, 2:] = self.sim_scaler.fit_transform(sim_scores[:, 2:])
        print("Done scaling")

        return sim_scores

    def cal_sim_score_knn_str(self, key1, key2, knn_k=3, psig_p=8):
        key1 = key1.astype('str').flatten()
        key2 = key2.astype('str').flatten()
        assert len(np.unique(key1)) == len(key1) and len(np.unique(key2)) == len(key2)

        key1_to_idx1 = {k: v for k, v in zip(key1, range(key1.shape[0]))}
        key2_to_idx2 = {k: v for k, v in zip(key2, range(key2.shape[0]))}

        # blocking
        block_dict1 = {}
        for title in key1:
            key = title[:psig_p]
            if key in block_dict1:
                block_dict1[key].append(title)
            else:
                block_dict1[key] = [title]

        block_dict2 = {}
        for title in key2:
            key = title[:psig_p]
            if key in block_dict2:
                block_dict2[key].append(title)
            else:
                block_dict2[key] = [title]

        print("#blocks in party 1: {}".format(len(block_dict1)))
        print("#blocks in party 2: {}".format(len(block_dict2)))

        # Compare
        title_sim_scores = {}
        for party1_key, party1_block in tqdm(block_dict1.items()):
            if party1_key not in block_dict2:
                continue

            party2_block = block_dict2[party1_key]
            for party1_title in party1_block:
                for party2_title in party2_block:
                    dist = scaled_edit_distance(party1_title, party2_title)
                    idx1 = key1_to_idx1[party1_title]
                    idx2 = key2_to_idx2[party2_title]

                    if idx1 not in title_sim_scores:
                        title_sim_scores[idx1] = PriorityQueue()
                    title_sim_scores[idx1].put((dist, idx2))

        dist_idx2_all = []
        for idx1 in range(key1.shape[0]):
            indices2_dist = [(1, -1) if idx1 not in title_sim_scores or
                                        title_sim_scores[idx1].empty() else
                             title_sim_scores[idx1].get() for _ in range(knn_k)]
            dist_idx2_all += indices2_dist
        idx2_dist_all = np.array(dist_idx2_all)[:, ::-1]

        idx1_all = np.repeat(np.arange(key1.shape[0]), knn_k).reshape(-1, 1)

        sim_scores = np.concatenate([idx1_all, idx2_dist_all], axis=1)
        sim_scores[:, 2] = 1 - sim_scores[:, 2]

        if self.sim_scaler is not None:
            # use train scaler
            sim_scores[:, 2:] = self.sim_scaler.transform(sim_scores[:, 2:])
        else:
            # generate train scaler
            self.sim_scaler = StandardScaler()
            sim_scores[:, 2:] = self.sim_scaler.fit_transform(sim_scores[:, 2:])
        print("Done scaling")

        return sim_scores

    def cal_sim_score_radius(self, key1, key2, radius=3, tree_leaf_size=40):
        """
        :param tree_leaf_size: leaf size to build kd-tree
        :param radius: only the distance with radius will be returned
        :param key1: Common features in party 1
        :param key2: Common features in party 2
        :return: sim_scores: Nx3 np.ndarray (idx_key1, idx_key2, sim_score)
        """
        print("Build KD-tree")
        tree = KDTree(key2, leaf_size=tree_leaf_size)

        print("Query KD-tree")
        idx2, dists = tree.query_radius(key1, r=radius, return_distance=True, sort_results=False)

        print("Calculate sim_scores")
        repeat_times = [x.shape[0] for x in idx2]
        # non_empty_idx = [x > 0 for x in repeat_times]
        idx1 = np.repeat(np.arange(key1.shape[0]), repeat_times)
        idx2 = np.concatenate(idx2[np.array(repeat_times) > 0])
        if self.feature_wise_sim:
            sims = -(key1[idx1] - key2[idx2]) ** 2
        else:
            sims = -np.concatenate(dists[np.array(repeat_times) > 0]).reshape(-1, 1)
        # assert np.isclose(sims, -np.sqrt(np.sum((key1[idx1] - key2[idx2]) ** 2, axis=1)))
        sim_scores = np.concatenate([idx1.reshape(-1, 1), idx2.reshape(-1, 1), sims], axis=1)
        if self.sim_scaler is not None:
            sim_scores[:, 2:] = self.sim_scaler.transform(sim_scores[:, 2:])
        else:
            self.sim_scaler = StandardScaler()
            sim_scores[:, 2:] = self.sim_scaler.fit_transform(sim_scores[:, 2:])
        print("Done scaling")

        return sim_scores

    def cal_sim_score_knn_priv(self, key1, key2, key_type, knn_k=3):
        """
        Calculate sim_scores based on the distance of bit-vectors/bloom-filters.
        Ref: FEDERAL: A Framework for Distance-Aware Privacy-Preserving Record Linkage
        :param key_type: Type of keys. Support str and float.
        :param key1: Common features in party 1
        :param key2: Common features in party 2
        :param knn_k: number of nearest neighbors to be calculated
        :return: sim_scores: Nx3 np.ndarray (idx_key1, idx_key2, sim_score)
        """
        if key_type == 'str':
            key1 = key1.astype('str')
            key2 = key2.astype('str')
            bf1_vecs, bf2_vecs = self.str_to_bloom_filter(key1, key2,
                                                          edit_distance_threshold=self.edit_distance_threshold,
                                                          n_hash_func=self.n_hash_func,
                                                          collision_rate=self.collision_rate,
                                                          qgram_q=self.qgram_q, delta=self.link_delta)
        elif key_type == 'float':
            bf1_vecs, bf2_vecs = self.float_to_bloom_filter(key1, key2,
                                                            dist_threshold_t=self.link_threshold_t,
                                                            epsilon=self.link_epsilon,
                                                            delta=self.link_delta)
        else:
            assert False, "Not Supported key type"

        # Debug: sparsity check
        # zeros_cols = np.all(np.isclose(bf1_vecs, 0), axis=0) & np.all(np.isclose(bf2_vecs, 0), axis=0)
        # ones_cols = np.all(np.isclose(bf1_vecs, 1), axis=0) & np.all(np.isclose(bf2_vecs, 1), axis=0)
        # sparsity = np.count_nonzero(zeros_cols | ones_cols) / bf1_vecs.shape[1]
        # print("Sparsity of bloom filters is {}".format(sparsity))

        # link the bloom filters of party 1 and 2
        # res = faiss.StandardGpuResources()
        ndim = bf2_vecs.shape[1] * 8
        nlist = 100
        index = faiss.IndexBinaryFlat(ndim)
        # index = faiss.IndexBinaryIVF(quantizier, ndim, nlist)
        # index.nprobe = 5

        # m = 10000
        # nlist = 100
        # quantizier = faiss.IndexFlatL2(bf2_vecs.shape[1])
        # index = faiss.IndexIVFPQ(quantizier, bf2_vecs.shape[1], nlist, m, 8)
        # index.nprobe = 5

        # gpu_index = faiss.index_cpu_to_gpu(res, self.device.index, index)
        # gpu_index = faiss.index_cpu_to_all_gpus(index, ngpu=7)
        gpu_index = index
        # print("Training index")
        # gpu_index.train(bf2_vecs)

        # Pycharm false warning
        # noinspection PyArgumentList
        print("Adding index")
        gpu_index.add(bf2_vecs)
        print("Indexing done. Got {} samples.".format(gpu_index.ntotal))

        print("KNN Query")
        dists = np.empty([bf1_vecs.shape[0], knn_k])
        idx2 = np.empty([bf1_vecs.shape[0], knn_k])
        batch_size = 2000
        for i in tqdm(range(0, bf1_vecs.shape[0], batch_size)):
            dists_i, idx2_i = gpu_index.search(bf1_vecs[i: i + batch_size], k=knn_k)
            dists[i: i + batch_size] = dists_i
            idx2[i: i + batch_size] = idx2_i
        print("Done")

        gpu_index.reset()

        print("Calculate sim_scores")
        repeat_times = [x.shape[0] for x in idx2]
        idx1 = np.repeat(np.arange(key1.shape[0]), repeat_times)
        idx2 = np.concatenate(idx2[np.array(repeat_times) > 0])
        if self.feature_wise_sim:
            sims = -(bf1_vecs[idx1] - bf2_vecs[idx2]) ** 2
        else:
            sims = -np.concatenate(dists[np.array(repeat_times) > 0]).reshape(-1, 1)

        sim_scores = np.concatenate([idx1.reshape(-1, 1), idx2.reshape(-1, 1), sims.astype('float64')], axis=1)

        # assert np.isclose(sims, -np.sqrt(np.sum((key1[idx1] - key2[idx2]) ** 2, axis=1)))

        print("Scaling sim scores")
        if self.sim_scaler is not None:
            sim_scores[:, 2:] = self.sim_scaler.transform(sim_scores[:, 2:])
        else:
            print("Saving raw sim scores")
            np.save('cache/sim_scores_test.npy', sim_scores)
            self.sim_scaler = StandardScaler()
            sim_scores[:, 2:] = self.sim_scaler.fit_transform(sim_scores[:, 2:])
        print("Done scaling")

        return sim_scores

    def str_to_bloom_filter(self, key1, key2, edit_distance_threshold=1, n_hash_func=10, collision_rate=0.05,
                            qgram_q=2, delta=0.1):
        # Get the expected number of qgrams of each record
        n_qgrams = 0
        for k in np.concatenate([key1, key2]):
            n = sum([len(s) - qgram_q + 1 for s in k])
            n_qgrams += n
        n_qgrams //= key1.shape[0] + key2.shape[0]

        # calculate the optimal size of bloom filter according to the paper
        collision_num = np.ceil(4 * edit_distance_threshold * n_hash_func * collision_rate)
        bf_size = np.ceil((2 * edit_distance_threshold * n_hash_func * n_qgrams) /
                          ((collision_num + 1) * delta)).astype('int')
        bf_size = int(np.ceil(bf_size / 8) * 8)  # Match binary index of faiss
        print("Size of bloom filter = {}".format(bf_size))

        bf1_vecs = np.empty([key1.shape[0], np.sum(bf_size) // 8], dtype=np.uint8)
        for i, k in enumerate(tqdm(key1, desc='BF of key1')):
            # Generate q-grams
            qgrams = [''.join(s) for s in list(ngrams(''.join(k), qgram_q))]

            # Generate bloom filter
            bf = BloomFilter(m=bf_size, k=n_hash_func)
            for gram in qgrams:
                bf.add(gram)
            bf1_vecs[i, :] = np.packbits(bf.vector, axis=-1).reshape(1, -1)

        bf2_vecs = np.empty([key2.shape[0], np.sum(bf_size) // 8],
                            dtype=np.uint8)  # Allocate memory in advance to accelerate
        for i, k in enumerate(tqdm(key2, desc='BF of key2')):
            # Generate q-grams
            qgrams = [''.join(s) for s in list(ngrams(''.join(k), qgram_q))]

            # Generate bloom filter
            bf = BloomFilter(m=bf_size, k=n_hash_func)
            for gram in qgrams:
                bf.add(gram)
            bf2_vecs[i, :] = np.packbits(bf.vector, axis=-1).reshape(1, -1)

        return bf1_vecs, bf2_vecs

    def float_to_bloom_filter(self, key1, key2, dist_threshold_t=0.1, epsilon=0.1, delta=0.1):
        key1_min, key1_max = np.min(key1, axis=0), np.max(key1, axis=0)
        key2_min, key2_max = np.min(key2, axis=0), np.max(key2, axis=0)

        key_min = np.minimum(key1_min, key2_min)
        key_max = np.maximum(key1_max, key2_max)

        key_range = key_max - key_min

        bv_size = np.ceil((2 * key_range * np.log(1 / delta)) / (dist_threshold_t * epsilon ** 2)).astype('int')
        bv_size = (np.ceil(bv_size / 8) * 8).astype('int')  # Match binary index of faiss
        print("Size of bit vectors = {}".format(bv_size))

        pivots = []
        for k_min, k_max, m in zip(key1_min, key1_max, bv_size):
            pivots.append(np.random.uniform(k_min, k_max, m))

        bv1_vecs = np.empty([key1.shape[0], np.sum(bv_size) // 8],
                            dtype=np.uint8)  # Allocate memory in advance to accelerate
        for i, k in enumerate(tqdm(key1, desc='BV of key1')):
            bv1_vec = np.empty(0)
            for num, pivot in zip(k, pivots):
                vec = (num >= pivot - dist_threshold_t) & (num <= pivot + dist_threshold_t)
                bv1_vec = np.concatenate([bv1_vec, np.packbits(vec, axis=-1)], axis=0)
                # bv1_vec = np.concatenate([bv1_vec, vec.astype('float32')], axis=0)
            bv1_vecs[i, :] = bv1_vec.reshape(1, -1)

        bv2_vecs = np.empty([key2.shape[0], np.sum(bv_size) // 8],
                            dtype=np.uint8)  # Allocate memory in advance to accelerate
        for i, k in enumerate(tqdm(key2, desc='BV of key2')):
            bv2_vec = np.empty(0)
            for num, pivot in zip(k, pivots):
                vec = (num >= pivot - dist_threshold_t) & (num <= pivot + dist_threshold_t)
                bv2_vec = np.concatenate([bv2_vec, np.packbits(vec, axis=-1)], axis=0)
                # bv2_vec = np.concatenate([bv2_vec, vec.astype('float32')], axis=0)
            bv2_vecs[i, :] = bv2_vec.reshape(1, -1)

        return bv1_vecs, bv2_vecs

    def match(self, data1, data2, labels, idx=None, preserve_key=False, sim_threshold=0.0, grid_min=-3., grid_max=3.01,
              grid_width=0.2, knn_k=3, tree_leaf_size=40, radius=0.1):
        """
        Match data1 and data2 according to common features
        :param radius:
        :param knn_k:
        :param tree_leaf_size:
        :param blocking_method: method of blocking before matching
        :param knn_k: number of nearest neighbors to be calculated
        :param tree_leaf_size: leaf size to build kd-tree
        :param idx: Index of training data1, not involved in linkage
        :param sim_threshold: sim_threshold: threshold of similarity score.
               Everything below the threshold will be removed
        :param data1: [<other features in party 1>, common_features]
        :param data2: [common_features, <other features in party 2>]
        :param labels: corresponding labels
        :param preserve_key:
        :return: [data1, data2], labels
                 data1 = [sim_score, <other features in party 1>]
                 data2 = [sim_score, <other features in party 2>]
        """
        # extract common features from data
        key1 = data1[:, -self.num_common_features:]
        key2 = data2[:, :self.num_common_features]

        # calculate similarity scores
        if self.blocking_method == 'grid':
            sim_scores = self.cal_sim_score_grid(key1, key2, grid_min=grid_min, grid_max=grid_max,
                                                 grid_width=grid_width)
        elif self.blocking_method == 'knn':
            sim_scores = self.cal_sim_score_knn(key1, key2, knn_k=knn_k, tree_leaf_size=tree_leaf_size)
        elif self.blocking_method == 'knn_str':
            sim_scores = self.cal_sim_score_knn_str(key1, key2, knn_k=knn_k, psig_p=self.psig_p)
        elif self.blocking_method == 'radius':
            sim_scores = self.cal_sim_score_radius(key1, key2, radius=radius, tree_leaf_size=tree_leaf_size)
        elif self.blocking_method == 'knn_priv_str':
            sim_scores = self.cal_sim_score_knn_priv(key1, key2, key_type='str', knn_k=knn_k)
        elif self.blocking_method == 'knn_priv_float':
            sim_scores = self.cal_sim_score_knn_priv(key1, key2, key_type='float', knn_k=knn_k)
        elif self.blocking_method == 'knn_he_float':
            sim_scores = self.cal_sim_score_knn_he(key1, key2, knn_k=knn_k, grid_min=grid_min, grid_max=grid_max,
                                                   grid_width=grid_width)
        else:
            assert False, "Unsupported blocking method"

        if preserve_key:
            remain_data1 = data1.astype('float32')
            remain_data2 = data2.astype('float32')
        else:
            remain_data1 = data1[:, :-self.num_common_features].astype('float32')
            remain_data2 = data2[:, self.num_common_features:].astype('float32')

        # real_sim_scores = []
        # for idx1, idx2, score in sim_scores:
        #     real_sim_scores.append([idx[int(idx1)], int(idx2), score])
        # real_sim_scores = np.array(real_sim_scores)
        real_sim_scores = np.concatenate([idx[sim_scores[:, 0].astype(np.int)].reshape(-1, 1),
                                          sim_scores[:, 1:]], axis=1)

        # # filter similarity scores (last column) by a threshold
        # if not self.feature_wise_sim:
        #     real_sim_scores = real_sim_scores[real_sim_scores[:, -1] >= sim_threshold]
        # elif not np.isclose(sim_threshold, 0.0):
        #     warnings.warn("Threshold is not used for feature-wise similarity")

        # # save sim scores
        # with open("cache/sim_scores_test.pkl", "wb") as f:
        #     pickle.dump(real_sim_scores, f)

        remain_data1_shape1 = remain_data1.shape[1]

        # convert to pandas
        data1_df = pd.DataFrame(remain_data1)
        data1_df['data1_idx'] = idx
        labels_df = pd.DataFrame(labels, columns=['y'])
        data2_df = pd.DataFrame(remain_data2)
        if self.feature_wise_sim:
            score_columns = ['score' + str(i) for i in range(self.num_common_features)]
        else:
            score_columns = ['score']
        sim_scores_df = pd.DataFrame(real_sim_scores, columns=['data1_idx', 'data2_idx'] + score_columns)
        sim_scores_df[['data1_idx', 'data2_idx']].astype('int32')
        data1_labels_df = pd.concat([data1_df, labels_df], axis=1)

        matched_pairs = np.unique(sim_scores_df['data1_idx'].to_numpy())
        print("Got {} samples in A".format(matched_pairs.shape[0]))

        del remain_data1, remain_data2, labels, matched_pairs
        gc.collect()

        print("Setting index")
        data1_labels_df.set_index('data1_idx', inplace=True)
        data2_df['data2_idx'] = range(len(data2_df.index))
        data2_df.set_index('data2_idx', inplace=True)
        # sim_scores_df.set_index(['data1_idx', 'data2_idx'], inplace=True)

        print("Linking records")
        data1_labels_scores_df = sim_scores_df.merge(data1_labels_df, how='right', on='data1_idx')
        print("Step 1 done.")
        del data1_labels_df, sim_scores_df
        gc.collect()
        merged_data_labels_df = data1_labels_scores_df.merge(data2_df, how='left', left_on='data2_idx',
                                                             right_index=True)
        print("Finished Linking, got {} samples".format(len(merged_data_labels_df.index)))
        del data2_df
        gc.collect()

        print("Filling null values")
        merged_data_labels_df.fillna({col: 0.0 for col in score_columns}, inplace=True)

        print("extracting data to numpy arrays")
        ordered_labels = merged_data_labels_df['y'].to_numpy()
        data1_indices = merged_data_labels_df['data1_idx'].to_numpy()
        data2_indices = merged_data_labels_df['data2_idx'].to_numpy()
        data_indices = np.vstack([data1_indices, data2_indices]).T
        merged_data_labels_df.drop(['y', 'data1_idx', 'data2_idx'], axis=1, inplace=True)
        merged_data_labels = merged_data_labels_df.to_numpy()
        del merged_data_labels_df
        gc.collect()
        # merged_data_labels: |sim_scores|data1|data2|
        sim_dim = self.num_common_features if self.feature_wise_sim else 1
        matched_data1 = merged_data_labels[:, :sim_dim + remain_data1_shape1]
        matched_data2 = np.concatenate([merged_data_labels[:, :sim_dim],  # sim scores
                                        merged_data_labels[:, sim_dim + remain_data1_shape1:]],
                                       axis=1)
        # matched_data2 = merged_data_labels[:, sim_dim + remain_data1_shape1:]

        return [matched_data1, matched_data2], ordered_labels, data_indices

    def prepare_train_combine(self, data1, data2, labels, data_cache_path=None, scale=False):
        if data_cache_path and os.path.isfile(data_cache_path):
            print("Loading data from cache")
            with open(data_cache_path, 'rb') as f:
                train_dataset, val_dataset, test_dataset, y_scaler, self.sim_scaler = pickle.load(f)
            print("Done")
        else:
            print("Splitting data")
            train_data1, val_data1, test_data1, train_labels, val_labels, test_labels, train_idx1, val_idx1, test_idx1 = \
                self.split_data(data1, labels, val_rate=self.val_rate, test_rate=self.test_rate)

            if self.dataset_type == 'syn':
                train_data2 = data2[train_idx1]
                val_data2 = data2[val_idx1]
                test_data2 = data2[test_idx1]
            elif self.dataset_type == 'real':
                train_data2 = data2
                val_data2 = data2
                test_data2 = data2
            else:
                assert False, "Not supported dataset type"
            print("Matching training set")
            self.sim_scaler = None  # scaler will fit train_Xs and transform val_Xs, test_Xs
            preserve_key = not self.drop_key
            train_Xs, train_y, train_idx = self.match(train_data1, train_data2, train_labels, idx=train_idx1,
                                                      preserve_key=preserve_key, grid_min=self.grid_min,
                                                      grid_max=self.grid_max, grid_width=self.grid_width,
                                                      knn_k=self.knn_k, tree_leaf_size=self.tree_leaf_size,
                                                      radius=self.tree_radius)
            assert self.sim_scaler is not None
            print("Matching validation set")
            val_Xs, val_y, val_idx = self.match(val_data1, val_data2, val_labels, idx=val_idx1,
                                                preserve_key=preserve_key, grid_min=self.grid_min,
                                                grid_max=self.grid_max, grid_width=self.grid_width, knn_k=self.knn_k,
                                                tree_leaf_size=self.tree_leaf_size, radius=self.tree_radius)
            assert self.sim_scaler is not None
            print("Matching test set")
            test_Xs, test_y, test_idx = self.match(test_data1, test_data2, test_labels, idx=test_idx1,
                                                   preserve_key=preserve_key, grid_min=self.grid_min,
                                                   grid_max=self.grid_max, grid_width=self.grid_width, knn_k=self.knn_k,
                                                   tree_leaf_size=self.tree_leaf_size, radius=self.tree_radius)

            for train_X, val_X, test_X in zip(train_Xs, val_Xs, test_Xs):
                print("Replace NaN with mean value")
                col_mean = np.nanmean(train_X, axis=0)
                train_indices = np.where(np.isnan(train_X))
                train_X[train_indices] = np.take(col_mean, train_indices[1])
                print("Train done.")
                val_indices = np.where(np.isnan(val_X))
                val_X[val_indices] = np.take(col_mean, val_indices[1])
                print("Validation done.")
                test_indices = np.where(np.isnan(test_X))
                test_X[test_indices] = np.take(col_mean, test_indices[1])
                print("Test done.")

                if scale:
                    sim_dim = self.num_common_features if self.feature_wise_sim else 1
                    print("Scaling X")
                    x_scaler = StandardScaler()
                    train_X[:, sim_dim:] = x_scaler.fit_transform(train_X[:, sim_dim:])
                    val_X[:, sim_dim:] = x_scaler.transform(val_X[:, sim_dim:])
                    test_X[:, sim_dim:] = x_scaler.transform(test_X[:, sim_dim:])
                    # train_X[:] = x_scaler.fit_transform(train_X)
                    # val_X[:] = x_scaler.transform(val_X)
                    # test_X[:] = x_scaler.transform(test_X)
                    print("Scale done.")

            y_scaler = None
            if scale and self.task == 'regression':
                print("Scaling y")
                y_scaler = MinMaxScaler(feature_range=(0, 1))
                train_y = y_scaler.fit_transform(train_y.reshape(-1, 1)).flatten()
                val_y = y_scaler.transform(val_y.reshape(-1, 1)).flatten()
                test_y = y_scaler.transform(test_y.reshape(-1, 1)).flatten()
                print("Scale done")

            sim_dim = self.num_common_features if self.feature_wise_sim else 1
            train_dataset = SimDataset(train_Xs[0], train_Xs[1], train_y, train_idx, sim_dim=sim_dim)
            val_dataset = SimDataset(val_Xs[0], val_Xs[1], val_y, val_idx, sim_dim=sim_dim)
            test_dataset = SimDataset(test_Xs[0], test_Xs[1], test_y, test_idx, sim_dim=sim_dim)

            if data_cache_path:
                print("Saving data to cache")
                with open(data_cache_path, 'wb') as f:
                    pickle.dump([train_dataset, val_dataset, test_dataset, y_scaler, self.sim_scaler], f)
                print("Saved")

        print("Calculating noise scale")
        if np.isclose(self.sim_leak_p, 1.0):
            noise_scale = 0.0
        else:
            sim_std = self.sim_scaler.scale_.item()
            print("Standard variance of sim_score: {:.2f}".format(sim_std))
            self.scale_analysis = SimNoiseScale(sim_std, sim_leak_p=self.sim_leak_p)
            noise_scale = self.scale_analysis.noise_scale

        train_dataset.add_noise_to_sim_(noise_scale)
        val_dataset.add_noise_to_sim_(noise_scale)
        test_dataset.add_noise_to_sim_(noise_scale)

        if self.filter_top_k is not None:
            print("Filter dataset to top {}".format(self.filter_top_k))
            train_dataset.filter_to_topk_dataset_(self.filter_top_k)
            val_dataset.filter_to_topk_dataset_(self.filter_top_k)
            test_dataset.filter_to_topk_dataset_(self.filter_top_k)

            self.knn_k = self.filter_top_k

        return train_dataset, val_dataset, test_dataset, y_scaler

    @staticmethod
    def var_collate_fn(batch):
        data = torch.cat([item[0] for item in batch], dim=0)
        labels = torch.stack([item[1] for item in batch])
        weights = torch.cat([item[2] for item in batch], dim=0)
        idx = torch.cat([item[3] for item in batch], dim=0)
        idx_unique = np.array([item[4] for item in batch], dtype=np.int)
        return data, labels, weights, idx, idx_unique

    def plot_model(self, model, input_dim, save_fig_path, dim_wise=False):
        """
        If the input dimension of the model is lower than 2, plot the figure of model.
        Otherwise, do nothing.
        """
        assert int(input_dim) == input_dim
        plt.rcParams["font.size"] = 16

        if input_dim == 1:
            x = np.arange(-3, 3, 0.01)
            x_tensor = torch.tensor(x).float().reshape(-1, 1).to(self.device)
            z = model(x_tensor).detach().cpu().numpy()
            assert ((0 <= z) & (z <= 1)).all(), "{}".format(z)
            plt.plot(x, z)
            plt.xlabel(r"Similarity $s_{ij}$")
            plt.ylabel(r"Weight $w_{ij}$")
            plt.tight_layout()
            plt.savefig(save_fig_path)
            plt.close()
            return

        xs_raw = [np.arange(-3, 3, 0.01) for _ in range(input_dim)]
        xs = np.meshgrid(*xs_raw)
        xs_tensor = torch.tensor(np.concatenate(
            [x.reshape(-1, 1) for x in xs], axis=1)).float().to(self.device)
        z = model(xs_tensor).detach().cpu().numpy().reshape(xs[0].shape)

        if dim_wise:
            raise NotImplementedError
        else:
            assert input_dim == 2, "Cannot plot high dimensional functions"
            assert len(xs) == 2

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(xs[0], xs[1], z, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
            plt.savefig(save_fig_path)

    def visualize_model(self, model, data, target, save_fig_path, sim_model=None, sim_dim=1):
        model.eval()
        baselines = torch.zeros(data.shape).to(self.device)

        plt.rcParams["font.size"] = 16

        # Get feature importance by integrated gradients
        if sim_model is None:
            ig = IntegratedGradients(model)
        else:
            model_with_sim = lambda x: model(x[:, :, sim_dim:] * sim_model(x[:, :, :sim_dim]))
            ig = IntegratedGradients(model_with_sim)
        attributions, delta = ig.attribute(data, baselines, target=target, return_convergence_delta=True)
        avg_attrs = torch.mean(attributions, dim=0).detach().cpu().numpy()

        if len(avg_attrs.shape) == 1:
            # plot 1d heat map for the attributes
            fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
            x = np.arange(0, data.shape[1])
            extent = [x[0] - (x[1] - x[0]) / 2., x[-1] + (x[1] - x[0]) / 2., 0, 1]
            ax1.imshow(avg_attrs.reshape(1, -1), cmap='plasma_r', aspect='auto', extent=extent)
            ax1.set_yticks([])
            ax1.set_xlim(extent[0], extent[1])

            ax2.plot(x, avg_attrs)
            plt.tight_layout()
            plt.savefig(save_fig_path)
            plt.close()
        elif len(avg_attrs.shape) == 2:
            # if sim_model is not None:
            #     # do not plot sim column
            #     avg_attrs = avg_attrs[:, sim_dim:]

            # plot 2d heat map for the attributes
            plt.imshow(avg_attrs, cmap=cm.get_cmap('cool'))
            plt.ylabel("Indices of rows")
            plt.xticks(color='w')
            plt.tight_layout()
            plt.savefig(save_fig_path)
            plt.close()
        else:
            assert False, "Wrong dimension of input"
