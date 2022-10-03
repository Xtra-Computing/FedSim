import numpy as np
from sklearn.metrics import mean_squared_error

from .base import BaseMetric


class RMSE(BaseMetric):
    def __init__(self):
        super().__init__('RMSE')
        self.worst = np.inf

    def __call__(self, pred, label):
        all_pred = pred.flatten()
        assert all_pred.shape == label.shape
        return np.sqrt(mean_squared_error(label, all_pred))
