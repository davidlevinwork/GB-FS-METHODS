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
    def __init__(self, alpha: float, epochs: int = 50):
        super().__init__()
        self.alpha = alpha
        self.epochs = epochs
        self.is_done = False
        self.final_medoids = []
        self.cluster_details = []

    def __str__(self):
        return f'Greedy ({ALPHA}={self.alpha})'

    def run(self, data_props: DataProps, graph: GraphObject, kmedoids: dict, k: int) -> tuple:
        start_time = time.time()
        self.counter = 0

        while self.counter < self.epochs:
            self.counter += 1

            minimal_cost = self._get_minimal_space_cost(labels=kmedoids['labels'],
                                                        feature_costs=data_props.feature_costs)
            if minimal_cost > self.budget:
                kmedoids = self._run_kmedoids(data=graph.reduced_matrix, k=k)
                continue

            self._set_cluster_details(kmedoids=kmedoids, data_props=data_props)
            self._run_greedy_iteration(graph=graph, data_props=data_props)
            if self.is_done:
                break

        if self.is_done:
            mss, cost = self._calculate_new_feature_space(graph=graph, data_props=data_props)
        else:
            mss, cost = 0, 0

        end_time = time.time()
        log_service.log(f'[Heuristic Clustering] : [Greedy Heuristic ({ALPHA}={self.alpha})] : '
                        f'Total run time (sec) for [{k}] value: [{round(end_time - start_time, 3)}]'.
                        encode("utf-8").decode("utf-8"))

        return mss, cost

    def _set_cluster_details(self, kmedoids: dict, data_props: DataProps):
        for medoid, medoid_loc in zip(kmedoids['medoids'], kmedoids['medoids loc']):
            cluster_label = kmedoids['labels'][medoid]
            cluster_features_idx = [idx for idx, feature in enumerate(kmedoids['labels']) if feature == cluster_label]
            cluster_features_name = [data_props.features[idx] for idx in cluster_features_idx]
            cluster_features_cost = [list(data_props.feature_costs.values())[idx] for idx in cluster_features_idx]
            cluster_total_cost = sum(feature for feature in cluster_features_cost)
            medoid_cost = list(data_props.feature_costs.values())[medoid]
            medoid_name = list(data_props.feature_costs.keys())[medoid]

            self.cluster_details.append({
                'cluster_label': cluster_label,
                'medoid': medoid,
                'medoid_loc': medoid_loc,
                'medoid_cost': medoid_cost,
                'medoid_name': medoid_name,
                'cluster_features_idx': cluster_features_idx,
                'cluster_features_name': cluster_features_name,
                'cluster_features_cost': cluster_features_cost,
                'cluster_total_cost': cluster_total_cost
            })

        self.cluster_details = sorted(self.cluster_details, key=lambda x: x['medoid_cost'], reverse=True)

    def _run_greedy_iteration(self, data_props: DataProps, graph: GraphObject):
        """
        Function purpose it to run a full greedy iteration, i.e. find a potential clustering results that will satisfy
        the budget constraints.
        """
        for idx, cluster in enumerate(self.cluster_details):
            medoids_cost = sum(medoid['medoid_cost'] for medoid in self.cluster_details)
            if medoids_cost <= self.budget:
                self.is_done = True
                return

            next_sum = self._run_forward(idx=idx)
            prev_sum = self._run_backward(idx=idx)
            new_budget = self.budget - next_sum - prev_sum
            new_medoid = self._select_new_medoid(cluster=cluster, new_budget=new_budget, graph=graph)

            if new_medoid[0] == float('inf'):
                print(f"==> ERROR NUMBER 1")
                continue

            if new_medoid[0] != cluster['medoid']:
                print(f"==> CHANGE")
                self._update_new_medoid(cluster_idx=idx, new_medoid=new_medoid, graph=graph, data_props=data_props)

    def _calculate_new_feature_space(self, graph: GraphObject, data_props: DataProps):
        medoids, labels = self._get_new_kmedoids(graph=graph)
        mss = get_silhouette_value(type='mss',
                                   labels=labels,
                                   centroids=medoids,
                                   X=graph.reduced_matrix)
        cost = self.get_features_cost(data_props=data_props,
                                      features=np.array([medoid['medoid'] for medoid in self.cluster_details]))

        return mss, cost

    def _get_new_kmedoids(self, graph: GraphObject) -> tuple:
        medoids_idx = [medoid['medoid'] for medoid in self.cluster_details]
        medoids = [graph.reduced_matrix[center] for center in medoids_idx]
        labels = []

        for feature in graph.reduced_matrix:
            # For each feature in the space, find the closest medoid
            closest_centroid_idx = np.argmin([distance.euclidean(feature, graph.reduced_matrix[medoid])
                                              for medoid in medoids_idx])
            labels.append(closest_centroid_idx)

        return np.array(medoids), np.array(labels)

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

        relevant_features = [(feat_idx, feat_cost) for feat_idx, feat_cost in
                             zip(cluster['cluster_features_idx'], cluster['cluster_features_cost'])
                             if feat_cost < new_budget]
        if len(relevant_features) == 0:
            print(f"==> ERROR NUMBER 2")

        for feature in relevant_features:
            score = self.get_cost(cluster=cluster,
                                  feature=feature,
                                  graph=graph)
            if score < best_score:
                best_score = score
                best_cost = feature[1]
                best_feature_idx = feature[0]
        return best_feature_idx, best_cost

    def get_cost(self, cluster: dict, feature: tuple, graph: GraphObject) -> float:
        feature_to_medoid_dist = distance.euclidean(graph.reduced_matrix[feature[0]],
                                                    graph.reduced_matrix[cluster['medoid']])

        # return self.alpha * feature[1] + (1 - self.alpha) * feature_to_medoid_dist
        return feature[1]

    @staticmethod
    def _get_minimal_space_cost(labels: list, feature_costs: dict) -> float:
        # Return the minimal cost according to the feature space (take the min feature from each cluster)
        # (It's different then take the k minimalistic features from the entire space)
        cheapest_features = defaultdict(lambda: (None, float('inf')))

        for feature, label in zip(feature_costs.keys(), labels):
            cost = feature_costs[feature]
            if cost < cheapest_features[label][1]:
                cheapest_features[label] = (feature, cost)

        return sum(cost for feature, cost in cheapest_features.values())

    @staticmethod
    def _run_kmedoids(data: np.ndarray, k: int) -> dict:
        kmedoids = KMedoids(init='k-medoids++', n_clusters=k, method='pam').fit(data)

        return {
            'labels': kmedoids.labels_,
            'medoids': kmedoids.medoid_indices_,
            'medoids loc': kmedoids.cluster_centers_
        }
