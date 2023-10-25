import numpy as np

from ..config import config
from .heuristics import GreedyHeuristic
from ..models import GraphObject, DataProps


class HeuristicClusteringService:
    def __init__(self):
        self.budget = config.budget_constraint.budget
        self.heuristic_methods = []

    def run(self, data_props: DataProps, graph: GraphObject, kmedoids: dict, silhouette: dict, k: int) -> dict:
        self._init_heuristic_methods()

        min_k_features_cost = sum(sorted(data_props.feature_costs.values())[:k])
        features_cost = self._get_features_cost(data_props=data_props, features=kmedoids['medoids'])

        # True when Greedy finds new legal solution
        is_new_features, new_labels, new_medoids, new_medoids_loc = False, None, None, None

        cost_to_value = {'MSS': features_cost}
        mss_to_value = {'MSS': silhouette['MSS']}

        for heuristic_method in self.heuristic_methods:
            method_str = f'{str(heuristic_method)} MSS'

            if min_k_features_cost > self.budget:
                # The given feature space cannot satisfy the given budget --> None values
                mss_to_value[method_str] = None
                cost_to_value[method_str] = None
            elif features_cost <= self.budget:
                # The given feature space satisfies the given budget --> use 'regular' MSS values
                mss_to_value[method_str] = mss_to_value['MSS']
                cost_to_value[method_str] = cost_to_value['MSS']
            else:
                # The given feature space doesn't satisfy the given budget, but it possible --> use heuristic
                results = heuristic_method.run(k=k,
                                               graph=graph,
                                               kmedoids=kmedoids,
                                               data_props=data_props)
                mss_to_value[method_str] = results['mss']
                cost_to_value[method_str] = results['cost']
                is_new_features = results['is_new_features']
                if is_new_features:
                    new_labels = results['new_labels']
                    new_medoids = results['new_medoids']
                    new_medoids_loc = results['new_medoids_loc']

        return {
            'mss': mss_to_value,
            'cost': cost_to_value,
            'new_labels': new_labels,
            'new_medoids': new_medoids,
            'new_medoids_loc': new_medoids_loc,
            'is_new_features': is_new_features,
        }

    def _init_heuristic_methods(self):
        self.heuristic_methods = [GreedyHeuristic(alpha=0.5)]

    @staticmethod
    def _get_features_cost(data_props: DataProps, features: np.ndarray) -> float:
        return sum(list(data_props.feature_costs.values())[feature] for feature in features)
