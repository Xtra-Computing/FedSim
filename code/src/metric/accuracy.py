import numpy as np

from .base import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__('Accuracy')
        self.worst = -1

    def __call__(self, pred, label):
        all_pred = pred.flatten()
        assert all_pred.shape == label.shape
        return np.count_nonzero(all_pred == label) / label.shape[0]
