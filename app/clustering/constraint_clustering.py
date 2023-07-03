import numpy as np
from scipy.spatial import distance
from itertools import combinations

from ..config import config
from ..models import GraphObject, DataProps
from .silhouette import get_silhouette_value


class ConstraintClusteringService:
    def __init__(self):
        pass

    def run(self, data_props: DataProps, graph: GraphObject, kmedoids: dict, silhouette: dict, k: int) -> tuple:
        """
        Cost = regular   = original features
             = heuristic = best result under constraints
             = naive     = best result without constraints
        ***
        MSS = regular   = original features (already calculated and exists in the silhouette dict)
            = heuristic = best result under constraints
            = naive     = best result without constraints
        """
        costs = {}
        features_cost = self._get_features_cost(data_props=data_props, features=kmedoids['features'])
        costs['Regular C.'] = features_cost

        # Naive baseline (the best MSS value without constraints ; get cost based on best MSS value)
        naive_mss, naive_cost = self._get_baseline_results(k=k,
                                                           graph=graph,
                                                           is_naive=True,
                                                           data_props=data_props)
        costs['Naive C.'] = naive_cost
        silhouette['Naive MSS'] = naive_mss

        # If the original features satisfy the constraints - the heuristic and the regular are the same
        if features_cost <= config.constraint_satisfaction.budget:
            costs['Heuristic C.'] = costs['Regular C.']
            silhouette['Heuristic MSS'] = silhouette['MSS']

        # The original features doesn't satisfy the budget constraint ; there are two options:
        # 1. The k original features didn't satisfy the budget constraint, because there isn't any feasible solution
        # 2. The k original features didn't satisfy the budget constraint, but there is a combination that can do so
        else:
            min_k_features_cost = sum(sorted(data_props.feature_costs.values())[:k])
            # Option #1: basically, for this case the heuristic results are meaningless
            if min_k_features_cost > config.constraint_satisfaction.budget:
                costs['Heuristic C.'] = min_k_features_cost
                silhouette['Heuristic MSS'] = silhouette['MSS']
            # Option #2: find the best combination of k features (based on the highest MSS)
            else:
                # Heuristic baseline (the best MSS value with constraints ; get cost based on best MSS value)
                heuristic_mss, heuristic_cost = self._get_baseline_results(k=k,
                                                                           graph=graph,
                                                                           is_naive=False,
                                                                           data_props=data_props)
                costs['Heuristic C.'] = heuristic_cost
                silhouette['Heuristic MSS'] = heuristic_mss

        return silhouette, costs

    def _get_baseline_results(self, data_props: DataProps, graph: GraphObject, k: int, is_naive: bool) -> tuple:
        # This function will return the results (cost & mss) of the baseline methods (not the "real heuristics")

        # Create a list of all the possible combinations of k features from the (data) original list of features
        all_combinations = list(combinations(list(range(data_props.n_features)), k))

        # Define which combination os relevant, based on the budget constraint defined (is_naive = no budget)
        if is_naive:
            relevant_combinations = all_combinations
        else:
            relevant_combinations = [combination for combination in all_combinations if
                                     sum(list(data_props.feature_costs.values())[feat] for feat in combination)
                                     <= config.constraint_satisfaction.budget]

        new_kmedoids = self._get_new_kmedoids(graph=graph,
                                              features_combination=relevant_combinations)

        heuristic_mss, heuristic_cost = self._calculate_baseline_results(graph=graph,
                                                                         data_props=data_props,
                                                                         kmedoids=new_kmedoids)
        return heuristic_mss, heuristic_cost

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
                heuristic_cost = self._get_features_cost(features=combination,
                                                         data_props=data_props)

        return heuristic_mss, heuristic_cost

    @staticmethod
    def _get_new_kmedoids(graph: GraphObject, features_combination: list) -> list:
        results = []

        for f_combination in features_combination:
            if f_combination == (9, 11):
                x = 6
            labels = []
            # Define the 'new' centers as the combination features
            centers = [graph.reduced_matrix[center] for center in f_combination]

            for feature in graph.reduced_matrix:
                # For each feature in the space, find the closest center in the current combination features
                closest_centroid_idx = np.argmin([distance.euclidean(feature, graph.reduced_matrix[center])
                                                  for center in f_combination])
                labels.append(closest_centroid_idx)
            # Result for each feature combination = (centroids [=current combination], labels for current combination)
            if len(set(labels)) == 1:
                x = 5
                labels = 6
            results.append((f_combination, np.array(centers), np.array(labels)))

        return results

    @staticmethod
    def _get_features_cost(data_props: DataProps, features: np.ndarray) -> float:
        try:
            x = sum(list(data_props.feature_costs.values())[feature] for feature in features)
            return x
        except Exception as ex:
            y = 5
            x = str(ex)
            return
