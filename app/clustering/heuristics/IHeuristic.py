import numpy as np
from abc import ABC, abstractmethod

from ...config import config
from ...models import GraphObject, DataProps


class IHeuristic(ABC):
    def __init__(self):
        super().__init__()
        self._counter = 0
        self._budget = config.constraint_satisfaction.budget

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def run(self, data_props: DataProps, graph: GraphObject, kmedoids: dict, k: int) -> dict:
        pass

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def counter(self) -> int:
        return self._counter

    @counter.setter
    def counter(self, value):
        if isinstance(value, int) and value >= 0:
            self._counter = value
        else:
            raise ValueError("Counter value must be a non-negative integer")

    @staticmethod
    def get_features_cost(data_props: DataProps, features: np.ndarray) -> float:
        return sum(list(data_props.feature_costs.values())[feature] for feature in features)
