import os
import sys
import numpy as np
import random
from sklearn.datasets import make_classification
import pickle


class TwoPartyClsMany2ManyGenerator:
    def __init__(self, num_samples, num_features_per_party: list, num_common_features: int,
                 common_feature_noise_scale=0.1, common_feature_noise_bias=0.0, n_informative=20, n_redundant=10,
                 n_clusters_per_class=3, class_sep=0.7, n_classes=2, seed=0):
        """
        :param num_samples: number of samples
        :param num_features_per_party: number of features on both party, including common features
        :param num_common_features: number of common features
        """
        self.common_feature_noise_bias = common_feature_noise_bias
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.n_clusters_per_class = n_clusters_per_class
        self.class_sep = class_sep
        self.n_classes = n_classes

        self.common_feature_noise_scale = common_feature_noise_scale
        self.num_samples = num_samples
        self.num_common_features = num_common_features
        self.num_features_per_party = num_features_per_party
        self.seeds = list(range(seed, seed + 3))
        self.syn_X = None
        self.syn_y = None
        self.syn_Xs = None

        assert min(self.num_features_per_party[0], self.num_features_per_party[1]) > self.num_common_features

    def get_global(self):
        if self.syn_X is not None and self.syn_y is not None:
            return self.syn_X, self.syn_y

        # generate global dataset
        total_features = sum(self.num_features_per_party) - self.num_common_features
        self.syn_X, self.syn_y = make_classification(n_samples=self.num_samples, n_features=total_features,
                                                     n_informative=self.n_informative, n_redundant=self.n_redundant,
                                                     n_clusters_per_class=self.n_clusters_per_class,
                                                     class_sep=self.class_sep, n_classes=self.n_classes, shift=None,
                                                     random_state=self.seeds[0])

        return self.syn_X, self.syn_y

    def get_parties(self):
        if self.syn_Xs is not None and self.syn_y is not None:
            return self.syn_Xs, self.syn_y

        if self.syn_X is not None and self.syn_y is not None:
            syn_X, syn_y = self.syn_X, self.syn_y
        else:
            syn_X, syn_y = self.get_global()

        assert self.num_common_features < self.n_informative

        X_noise = syn_X[:, self.n_informative + self.n_redundant:]
        X_useful = syn_X[:, :self.n_informative + self.n_redundant]

        # randomly divide trained features to two parties
        trained_features = np.concatenate([X_noise, X_useful[:, self.num_common_features:]], axis=1)
        shuffle_state = np.random.RandomState(self.seeds[1])
        shuffle_state.shuffle(trained_features.T)  # shuffle columns
        trained_features1 = trained_features[:, :trained_features.shape[1] // 2]
        trained_features2 = trained_features[:, trained_features.shape[1] // 2:]

        # append common features
        common_features = X_useful[:, :self.num_common_features]
        noise_state = np.random.RandomState(self.seeds[2])
        noised_common_features = common_features.copy() + self.common_feature_noise_bias + \
                                 noise_state.normal(scale=self.common_feature_noise_scale, size=common_features.shape)
        syn_X1 = np.concatenate([trained_features1, common_features], axis=1)
        syn_X2 = np.concatenate([noised_common_features, trained_features2], axis=1)

        self.syn_Xs = [syn_X1, syn_X2]
        assert syn_X1.shape[1], syn_X2.shape[1] == self.num_features_per_party
        return self.syn_Xs, syn_y

    def to_pickle(self, save_path: str):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(load_path: str):
        with open(load_path, 'rb') as f:
            return pickle.load(f)


class TwoPartyClsOne2OneGenerator:
    def __init__(self, num_samples, num_features_per_party: list, num_common_features: int,
                 common_feature_noise_scale=0.1, seed=0):
        """
        :param num_samples: number of samples
        :param num_features_per_party: number of features on both party, including common features
        :param num_common_features: number of common features
        """
        random.seed(seed)

        self.common_feature_noise_scale = common_feature_noise_scale
        self.num_samples = num_samples
        self.num_common_features = num_common_features
        self.num_features_per_party = num_features_per_party
        self.seeds = list(range(seed, seed + 4))
        self.syn_Xs = None
        self.syn_y = None

        assert min(self.num_features_per_party[0], self.num_features_per_party[1]) > self.num_common_features

    def get_parties(self):
        if self.syn_Xs is not None and self.syn_y is not None:
            return self.syn_Xs, self.syn_y

        syn_X1, syn_y1 = make_classification(n_samples=self.num_samples,
                                             n_features=self.num_features_per_party[0] - self.num_common_features,
                                             n_informative=30, n_redundant=10, n_clusters_per_class=2,
                                             class_sep=0.9, n_classes=2, shift=None,
                                             random_state=self.seeds[0])
        syn_X2, syn_y2 = make_classification(n_samples=self.num_samples,
                                             n_features=self.num_features_per_party[1] - self.num_common_features,
                                             n_informative=30, n_redundant=10, n_clusters_per_class=2,
                                             class_sep=0.9, n_classes=2, shift=None,
                                             random_state=self.seeds[1])
        random_state1 = np.random.RandomState(self.seeds[2])
        random_state2 = np.random.RandomState(self.seeds[3])
        common_keys = random_state1.normal(scale=1.0, size=(self.num_samples, self.num_common_features))
        noised_common_keys = common_keys + random_state2.normal(scale=self.common_feature_noise_scale,
                                                                size=common_keys.shape)
        syn_X1 = np.concatenate([syn_X1, common_keys], axis=1)
        syn_X2 = np.concatenate([noised_common_keys, syn_X2], axis=1)

        # linear combination to get final label
        self.syn_y = (syn_y1 != syn_y2).astype(int)  # xor
        self.syn_Xs = [syn_X1, syn_X2]

        return self.syn_Xs, self.syn_y

    def to_pickle(self, save_path: str):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(load_path: str):
        with open(load_path, 'rb') as f:
            return pickle.load(f)



class TwoPartyOne2OneLinearGenerator:
    def __init__(self, num_samples, num_features_per_party: list, num_common_features: int,
                 common_feature_noise_bias=0.0, n_informative=20, n_redundant=10,
                 n_clusters_per_class=3, class_sep=0.7, n_classes=2, seed=0):
        """
        :param num_samples: number of samples
        :param num_features_per_party: number of features on both party, including common features
        :param num_common_features: number of common features
        """
        self.common_feature_noise_bias = common_feature_noise_bias
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.n_clusters_per_class = n_clusters_per_class
        self.class_sep = class_sep
        self.n_classes = n_classes

        self.num_samples = num_samples
        self.num_common_features = num_common_features
        self.num_features_per_party = num_features_per_party
        self.seeds = list(range(seed, seed + 3))
        self.syn_X = None
        self.syn_y = None
        self.syn_Xs = None

        assert min(self.num_features_per_party[0], self.num_features_per_party[1]) > self.num_common_features

    def get_global(self):
        if self.syn_X is not None and self.syn_y is not None:
            return self.syn_X, self.syn_y

        # generate global dataset
        total_features = sum(self.num_features_per_party) - self.num_common_features
        self.syn_X, self.syn_y = make_classification(n_samples=self.num_samples, n_features=total_features,
                                                     n_informative=self.n_informative, n_redundant=self.n_redundant,
                                                     n_clusters_per_class=self.n_clusters_per_class,
                                                     class_sep=self.class_sep, n_classes=self.n_classes, shift=None,
                                                     random_state=self.seeds[0])

        return self.syn_X, self.syn_y

    def get_parties(self):
        if self.syn_Xs is not None and self.syn_y is not None:
            return self.syn_Xs, self.syn_y

        if self.syn_X is not None and self.syn_y is not None:
            syn_X, syn_y = self.syn_X, self.syn_y
        else:
            syn_X, syn_y = self.get_global()

        assert self.num_common_features < self.n_informative

        X_noise = syn_X[:, self.n_informative + self.n_redundant:]
        X_useful = syn_X[:, :self.n_informative + self.n_redundant]

        # randomly divide trained features to two parties
        trained_features = np.concatenate([X_noise[:, self.num_common_features:], X_useful], axis=1)
        shuffle_state = np.random.RandomState(self.seeds[1])
        shuffle_state.shuffle(trained_features.T)  # shuffle columns
        trained_features1 = trained_features[:, :trained_features.shape[1] // 2]
        trained_features2 = trained_features[:, trained_features.shape[1] // 2:]

        # append common features
        common_features = X_noise[:, :self.num_common_features]
        # noise_state = np.random.RandomState(self.seeds[2])
        # noised_common_features = common_features.copy() + self.common_feature_noise_bias + \
        #                          noise_state.normal(scale=self.common_feature_noise_scale, size=common_features.shape)
        syn_X1 = np.concatenate([trained_features1, common_features], axis=1)
        syn_X2 = np.concatenate([common_features, trained_features2], axis=1)

        self.syn_Xs = [syn_X1, syn_X2]
        assert syn_X1.shape[1], syn_X2.shape[1] == self.num_features_per_party
        return self.syn_Xs, syn_y

    def to_pickle(self, save_path: str):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(load_path: str):
        with open(load_path, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    os.chdir(sys.path[0] + "/../../")  # change working directory
    root = "data/"
    syn_generator = TwoPartyClsOne2OneGenerator(num_samples=60000,
                                                num_features_per_party=[100, 100],
                                                num_common_features=3,
                                                common_feature_noise_scale=0.01)
    [X1, X2], y = syn_generator.get_parties()
    syn_generator.to_pickle(root + "syn_cls_one2one_generator.pkl")
