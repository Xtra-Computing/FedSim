import numpy as np
from sklearn.metrics import mean_absolute_error

from .base import BaseMetric


class MAE(BaseMetric):
    def __init__(self):
        super().__init__("MAE")
        self.worst = np.inf

    def __call__(self, pred, label):
        all_pred = pred.flatten()
        assert all_pred.shape == label.shape
        return mean_absolute_error(label, all_pred)
