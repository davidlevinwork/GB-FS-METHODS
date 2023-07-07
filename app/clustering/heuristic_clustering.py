import numpy as np

from ..config import config
from ..services import log_service
from ..models import GraphObject, DataProps
from .heuristics import BasicHeuristic, GreedyHeuristic


class HeuristicClusteringService:
    def __init__(self):
        self.budget = config.constraint_satisfaction.budget
        self.heuristic_methods = []

    def run(self, data_props: DataProps, graph: GraphObject, kmedoids: dict, silhouette: dict, k: int) -> tuple:
        self._init_heuristic_methods()

        min_k_features_cost = sum(sorted(data_props.feature_costs.values())[:k])
        features_cost = self._get_features_cost(data_props=data_props, features=kmedoids['medoids'])

        cost_to_value = {'MSS': features_cost}
        mss_to_value = {'MSS': silhouette['MSS']}

        for heuristic_method in self.heuristic_methods:
            if min_k_features_cost > self.budget or features_cost <= self.budget:
                mss_to_value[f'{str(heuristic_method)} MSS'] = 0                   # maybe none is better?
                cost_to_value[f'{str(heuristic_method)} MSS'] = 0                  # how we handle none in plot?
            else:
                mss, cost = heuristic_method.run(k=k,
                                                 graph=graph,
                                                 kmedoids=kmedoids,
                                                 data_props=data_props)
                mss_to_value[f'{str(heuristic_method)} MSS'] = mss
                cost_to_value[f'{str(heuristic_method)} MSS'] = cost

        # ==> ADD FUNCTION ==> self._get_minimum_results()
        return mss_to_value, cost_to_value

    def _init_heuristic_methods(self):
        self.heuristic_methods = [GreedyHeuristic(alpha=0.5),
                                  BasicHeuristic(is_naive=True),
                                  BasicHeuristic(is_naive=False)]

    @staticmethod
    def _get_features_cost(data_props: DataProps, features: np.ndarray) -> float:
        return sum(list(data_props.feature_costs.values())[feature] for feature in features)
