import time
import numpy as np
from scipy.spatial import distance
from collections import defaultdict
from sklearn_extra.cluster import KMedoids

from .IHeuristic import IHeuristic
from ...services import log_service
from ...models import GraphObject, DataProps
from ..silhouette import get_silhouette_value

ALPHA = r'$\alpha$'


class GreedyHeuristic(IHeuristic):
    def __init__(self, alpha: float, epochs: int = 100):
        super().__init__()
        self.alpha = alpha
        self.epochs = epochs
        self.is_done = False
        self.final_medoids = []
        self.cluster_details = []

    def __str__(self):
        return f'Greedy ({ALPHA}={self.alpha})'

    def run(self, data_props: DataProps, graph: GraphObject, kmedoids: dict, k: int) -> dict:
        start_time = time.time()
        self.counter = 0

        while self.counter < self.epochs:
            self.counter += 1

            minimal_cost = self._get_minimal_space_cost(labels=kmedoids['labels'],
                                                        feature_costs=data_props.feature_costs)
            if minimal_cost > self.budget:
                # If the current feature space cannot obtain a 'legal' results in terms of budget -
                # we generate new run of k-medoids
                kmedoids = self._run_kmedoids(data=graph.reduced_matrix, k=k)
                continue

            self._set_cluster_details(kmedoids=kmedoids, graph=graph, data_props=data_props)
            self._run_greedy_iteration(k=k, graph=graph, data_props=data_props)
            if self.is_done:
                break

        if self.is_done:
            final_results = self._calculate_new_feature_space(graph=graph, data_props=data_props)
            results = {
                'is_new_features': True,
                'mss': final_results['mss'],
                'cost': final_results['cost'],
                'new_labels': final_results['new_labels'],
                'new_medoids': final_results['new_medoids'],
                'new_medoids_loc': final_results['new_medoids_loc']
            }
        else:
            results = {
                'is_new_features': False,
                'mss': -1,
                'cost': -1
            }

        end_time = time.time()
        log_service.log(f'[Heuristic Clustering] : [Greedy Heuristic ({ALPHA}={self.alpha})] : '
                        f'For k=[{k}] => Succeeded: [{self.is_done}] ; Total number of iterations: [{self.counter}] ; '
                        f'Total run time (sec): [{round(end_time - start_time, 3)}].')
        return results

    def _set_cluster_details(self, kmedoids: dict, graph: GraphObject, data_props: DataProps):
        clusters = []
        for medoid, medoid_loc in zip(kmedoids['medoids'], kmedoids['medoid_loc']):
            cluster_label = kmedoids['labels'][medoid]
            cluster_features_idx = [idx for idx, feature in enumerate(kmedoids['labels']) if feature == cluster_label]
            cluster_features_name = [data_props.features[idx] for idx in cluster_features_idx]
            cluster_features_cost = [list(data_props.feature_costs.values())[idx] for idx in cluster_features_idx]
            cluster_total_cost = sum(feature for feature in cluster_features_cost)
            medoid_cost = list(data_props.feature_costs.values())[medoid]
            medoid_name = list(data_props.feature_costs.keys())[medoid]
            min_dist, max_dist = self._get_cluster_distance(graph=graph,
                                                            medoid=medoid,
                                                            features_idx=cluster_features_idx)
            clusters.append({
                'cluster_label': cluster_label,
                'medoid': medoid,
                'medoid_loc': medoid_loc,
                'medoid_cost': medoid_cost,
                'medoid_name': medoid_name,
                'cluster_features_idx': cluster_features_idx,
                'cluster_features_name': cluster_features_name,
                'cluster_features_cost': cluster_features_cost,
                'cluster_total_cost': cluster_total_cost,
                'cluster_distances': {
                    'max_dist': max_dist,
                    'min_dist': min_dist
                }
            })

        self.cluster_details = sorted(clusters, key=lambda x: x['medoid_cost'], reverse=True)

    @staticmethod
    def _get_cluster_distance(graph: GraphObject, medoid: int, features_idx: list):
        if len(features_idx) == 1:
            min_dist = 0
            max_dist = 0
        else:
            min_dist = sorted([distance.euclidean(graph.reduced_matrix[feature_idx], graph.reduced_matrix[medoid])
                               for feature_idx in features_idx])[1]

            max_dist = max([distance.euclidean(graph.reduced_matrix[feature_idx], graph.reduced_matrix[medoid])
                            for feature_idx in features_idx])

        return min_dist, max_dist

    @staticmethod
    def _get_minimal_space_cost(labels: list, feature_costs: dict) -> float:
        # Return the minimal cost according to the feature space (take the min feature from each cluster)
        # (It's different from take the k minimalistic features from the entire space)
        cheapest_features = defaultdict(lambda: (None, float('inf')))

        for feature, label in zip(feature_costs.keys(), labels):
            cost = feature_costs[feature]
            if cost < cheapest_features[label][1]:
                cheapest_features[label] = (feature, cost)

        return sum(cost for feature, cost in cheapest_features.values())

    def _run_greedy_iteration(self, k: int, data_props: DataProps, graph: GraphObject):
        """
        Function purpose it to run a full greedy iteration, i.e. find a potential clustering results that will satisfy
        the budget constraints.
        """
        for idx, cluster in enumerate(self.cluster_details):
            medoids_cost = sum(medoid['medoid_cost'] for medoid in self.cluster_details)
            if medoids_cost <= self.budget:
                self.is_done = True
                return

            # Clusters of size 1 won't change...
            if len(cluster['cluster_features_idx']) == 1:
                continue

            next_sum = self._run_forward(idx=idx)
            prev_sum = self._run_backward(idx=idx)
            new_budget = self.budget - next_sum - prev_sum
            new_medoid = self._select_new_medoid(cluster=cluster, new_budget=new_budget, graph=graph)

            if new_medoid[0] == float('inf'):
                log_service.log(f'[Heuristic Clustering] : [Greedy Heuristic ({ALPHA}={self.alpha})] : '
                                f'The method didnt succeeded to find any medoid for cluster number [{idx}] ; '
                                f'Original budget: [{self.budget}] ; New budget: [{new_budget}]',
                                level="Error")
                continue

            if new_medoid[0] != cluster['medoid']:
                log_service.log(f'[Heuristic Clustering] : [Greedy Heuristic ({ALPHA}={self.alpha})] : '
                                f'For k=[{k}] clusters, the method found a new medoid for cluster [{idx}] ; '
                                f'Old medoid: [{cluster["medoid"]}], New medoid: [{new_medoid[0]}].')
                self._update_new_medoid(cluster_idx=idx, new_medoid=new_medoid, graph=graph, data_props=data_props)

    def _update_new_medoid(self, graph: GraphObject, data_props: DataProps, cluster_idx: int, new_medoid: tuple):
        self.cluster_details[cluster_idx].update({
            'medoid': new_medoid[0],
            'medoid_loc': graph.reduced_matrix[new_medoid[0]],
            'medoid_cost': new_medoid[1],
            'medoid_name': data_props.features[new_medoid[0]]
        })

    def _run_backward(self, idx: int) -> float:
        return sum(item['medoid_cost'] for item in self.cluster_details[:idx])

    def _run_forward(self, idx: int) -> float:
        return sum(min(res['cluster_features_cost']) for res in self.cluster_details[idx + 1:])

    def _select_new_medoid(self, cluster: dict, new_budget: float, graph: GraphObject):
        # Cost & Index are changing based on the best_score
        best_score = float('inf')
        best_cost = float('inf')
        best_feature_idx = float('inf')

        special_dist = False
        if len(cluster['cluster_features_idx']) == 2:
            special_dist = True

        relevant_features = [(feat_idx, feat_cost) for feat_idx, feat_cost in
                             zip(cluster['cluster_features_idx'], cluster['cluster_features_cost'])
                             if feat_cost < new_budget]

        for feature in relevant_features:
            score = self._get_cost(graph=graph,
                                   cluster=cluster,
                                   feature=feature,
                                   special_dist=special_dist)
            if score < best_score:
                best_score = score
                best_cost = feature[1]
                best_feature_idx = feature[0]
        if best_score == float('inf') or best_cost == float('inf') or best_feature_idx == float('inf'):
            raise ValueError("Function [_select_new_medoids] didnt succeeded to find new medoid.")
        return best_feature_idx, best_cost

    def _get_cost(self, cluster: dict, feature: tuple, graph: GraphObject, special_dist: bool) -> float:
        feature_to_medoid_dist = distance.euclidean(graph.reduced_matrix[feature[0]],
                                                    graph.reduced_matrix[cluster['medoid']])

        if feature[0] == cluster['medoid'] or special_dist:
            norm_distance = 0
        else:
            norm_distance = np.divide(feature_to_medoid_dist - cluster['cluster_distances']['min_dist'],
                                      cluster['cluster_distances']['max_dist'] - cluster['cluster_distances']['min_dist'])

        norm_cost = np.divide(feature[1] - min(cluster['cluster_features_cost']),
                              max(cluster['cluster_features_cost']) - min(cluster['cluster_features_cost']))

        if self.alpha * norm_cost + (1 - self.alpha) * norm_distance < 0:
            print("==> NEGATIVE VALUE")

        return self.alpha * norm_cost + (1 - self.alpha) * norm_distance

    @staticmethod
    def _run_kmedoids(data: np.ndarray, k: int) -> dict:
        kmedoids = KMedoids(init='k-medoids++', n_clusters=k, method='pam').fit(data)

        return {
            'labels': kmedoids.labels_,
            'medoids': kmedoids.medoid_indices_,
            'medoid_loc': kmedoids.cluster_centers_
        }

    def _calculate_new_feature_space(self, graph: GraphObject, data_props: DataProps) -> dict:
        labels, medoids, medoids_loc = self._get_new_kmedoids(graph=graph)
        mss = get_silhouette_value(type='mss',
                                   labels=labels,
                                   centroids=medoids_loc,
                                   X=graph.reduced_matrix)
        cost = self.get_features_cost(data_props=data_props, features=medoids)

        return {
            'mss': mss,
            'cost': cost,
            'new_labels': labels,
            'new_medoids': medoids,
            'new_medoids_loc': medoids_loc
        }

    def _get_new_kmedoids(self, graph: GraphObject) -> tuple:
        medoids_idx = [medoid['medoid'] for medoid in self.cluster_details]
        medoids = [graph.reduced_matrix[center] for center in medoids_idx]
        labels = []

        for feature in graph.reduced_matrix:
            # For each feature in the space, find the closest medoid
            closest_centroid_idx = np.argmin([distance.euclidean(feature, graph.reduced_matrix[medoid])
                                              for medoid in medoids_idx])
            labels.append(closest_centroid_idx)

        return np.array(labels), np.array(medoids_idx), np.array(medoids)
