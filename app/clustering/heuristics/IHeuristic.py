from abc import ABC, abstractmethod

from ...models import GraphObject, DataProps


class IHeuristic(ABC):
    def __init__(self):
        super().__init__()
        self._counter = 0

    @abstractmethod
    def run(self, data_props: DataProps, graph: GraphObject, kmedoids: dict, k: int):
        pass

    def get_cost(self):
        return 0

    @property
    def counter(self):
        return self._counter

    @counter.setter
    def counter(self, value):
        if isinstance(value, int) and value >= 0:
            self._counter = value
        else:
            raise ValueError("Counter value must be a non-negative integer")
