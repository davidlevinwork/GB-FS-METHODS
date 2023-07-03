import time
import numpy as np

from .IHeuristic import IHeuristic
from ...models import GraphObject, DataProps
from ...services.log_service import log_service


class GreedyHeuristic(IHeuristic):
    def __init__(self):
        super().__init__()

    def run(self, data_props: DataProps, graph: GraphObject, kmedoids: dict, k: int):
        start_time = time.time()

        end_time = time.time()

    @staticmethod
    def _sort_medoids_by_cost(feature_costs: dict, centers: np.ndarray) -> list:
        feature_names = list(feature_costs.keys())
        indexed_feature_costs = {i: (feature, feature_costs[feature]) for i, feature in enumerate(feature_names)}

        # Filter the dictionary
        filtered_feature_costs = {i: cost for i, cost in indexed_feature_costs.items() if i in centers}

        # Sort the dictionary by value in descending order and extract the keys
        sorted_features = sorted(filtered_feature_costs, key=lambda x: filtered_feature_costs[x][1], reverse=True)

        # Map sorted indices back to feature names and costs: (feature name, feature index, feature cost)
        sorted_feature_tuples = [(filtered_feature_costs[i][0], i, filtered_feature_costs[i][1]) for i in
                                 sorted_features]

        return sorted_feature_tuples
