import numpy as np
from sklearn.metrics import r2_score

from .base import BaseMetric


class R2Score(BaseMetric):
    def __init__(self):
        super().__init__('R2_Score')
        self.worst = -np.inf

    def __call__(self, pred, label):
        all_pred = pred.flatten()
        assert all_pred.shape == label.shape
        return r2_score(label, all_pred)
