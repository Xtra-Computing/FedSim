import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitNN(nn.Module):
    def __init__(self, local_models, local_input_dims, agg_model):
        super().__init__()
        self.local_input_dims = local_input_dims
        self.agg_model = agg_model
        self.local_models = nn.ModuleList(local_models)

        self.n_local_models = len(self.local_models)
        assert self.n_local_models == len(self.local_input_dims)

    @staticmethod
    def split_features(x, dims: list):
        """
        split features of x according to dims
        :param x: two-dimensional matrix
        :param dims: list of int
        :return:
        """
        assert sum(dims) == x.shape[1]
        xs = []
        s_pos = 0
        for dim in dims:
            xs.append(x[:, s_pos: s_pos+dim])
            s_pos += dim
        return xs

    def forward(self, X):
        Xs = self.split_features(X, self.local_input_dims)
        local_out = torch.cat([self.local_models[i](Xs[i])
                               for i in range(self.n_local_models)], dim=1)
        out = self.agg_model(local_out)
        return out
