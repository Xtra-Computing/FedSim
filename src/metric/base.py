import abc


class BaseMetric(abc.ABC):
    def __init__(self, name):
        self.name = name
        self.worst = None

    @abc.abstractmethod
    def __call__(self, pred, label) -> float:
        """
        Calculate score for pred and label
        :param pred: Size (n_samples, n_pred_dim)
        :param label: Size (n_samples)
        :return: Score
        """
        pass

