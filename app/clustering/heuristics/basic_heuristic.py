import time
import numpy as np
from scipy.spatial import distance
from itertools import combinations

from ...config import config
from .IHeuristic import IHeuristic
from ...services import log_service
from ...models import GraphObject, DataProps
from ..silhouette import get_silhouette_value


class BasicHeuristic(IHeuristic):
    def __init__(self, is_naive: bool):
        super().__init__()
        self.is_naive = is_naive

    def __str__(self):
        return 'Basic Naive' if self.is_naive else 'Basic No Naive'

    def run(self, data_props: DataProps, graph: GraphObject, kmedoids: dict, k: int):
        start_time = time.time()

        mss, cost = self._get_results(k=k,
                                      graph=graph,
                                      data_props=data_props)

        end_time = time.time()
        log_service.log(f'[Heuristic Clustering] : [Basic Heuristic (Naive mode = {self.is_naive})] : '
                        f'Total run time (sec) for [{k}] value: [{round(end_time - start_time, 3)}]')

        return mss, cost

    def _get_results(self, data_props: DataProps, graph: GraphObject, k: int) -> tuple:
        # Create a list of all the possible combinations of k features from the (data) original list of feature
        all_combinations = list(combinations(list(range(data_props.n_features)), k))

        # Define which combination os relevant, based on the budget constraint defined (is_naive = no budget)
        if self.is_naive:
            relevant_combinations = all_combinations
        else:
            relevant_combinations = [combination for combination in all_combinations if
                                     sum(list(data_props.feature_costs.values())[feat] for feat in combination)
                                     <= config.constraint_satisfaction.budget]

        new_kmedoids = self._get_new_kmedoids(graph=graph,
                                              feature_combinations=relevant_combinations)
        mss, cost = self._calculate_baseline_results(graph=graph,
                                                     data_props=data_props,
                                                     kmedoids=new_kmedoids)
        return mss, cost

    @staticmethod
    def _get_new_kmedoids(graph: GraphObject, feature_combinations: list) -> list:
        results = []
        for feature_combination in feature_combinations:
            centers = graph.reduced_matrix[list(feature_combination)]
            dist_matrix = distance.cdist(graph.reduced_matrix, centers)
            labels = np.argmin(dist_matrix, axis=1)
            results.append((feature_combination, centers, labels))
        return results

    def _calculate_baseline_results(self, data_props: DataProps, graph: GraphObject, kmedoids: list) -> tuple:
        heuristic_mss = 0
        heuristic_cost = -1

        for combination, centroids, labels in kmedoids:
            mss = get_silhouette_value(type='mss',
                                       labels=labels,
                                       centroids=centroids,
                                       X=graph.reduced_matrix)
            if mss >= heuristic_mss:
                heuristic_mss = mss
                heuristic_cost = self.get_features_cost(features=combination,
                                                        data_props=data_props)

        return heuristic_mss, heuristic_cost
