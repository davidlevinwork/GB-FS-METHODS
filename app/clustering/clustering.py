import os
import time
import numpy as np
import concurrent.futures
from sklearn_extra.cluster import KMedoids

from ..config import config
from ..services import log_service
from .silhouette import get_silhouette_value
from ..models import GraphObject, DataProps, OPERATION_MODE
from .heuristic_clustering import HeuristicClusteringService
from ..services.plot_service import plot_silhouette, plot_clustering, plot_jm_clustering, plot_costs_to_silhouette


class ClusteringService:
    def __init__(self, max_workers: int = None):
        self.heuristic_service = HeuristicClusteringService()
        self.max_workers = max_workers or min(32, os.cpu_count() + 4)

    def run(self, data_props: DataProps, graph: GraphObject, k_range: list, stage: str, fold_index: int) -> list:
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            tasks = [executor.submit(self._run_cluster_evaluation, data_props, graph, k) for k in k_range]
            results = [task.result() for task in concurrent.futures.as_completed(tasks)]

        results = self._log_results(results=results)
        self._plot_plots(stage=stage, graph=graph, results=results, fold_index=fold_index)

        end_time = time.time()
        log_service.log(f'[Clustering Service] : Total run time (sec): [{round(end_time - start_time, 3)}]')

        return results

    def _run_cluster_evaluation(self, data_props: DataProps, graph: GraphObject, k: int) -> dict:
        kmedoids = self._run_kmedoids(data=graph.reduced_matrix, k=k)
        silhouette = self._get_silhouette_value(data=graph.reduced_matrix,
                                                labels=kmedoids['labels'],
                                                centroids=kmedoids['medoids loc'])
        result = {
            'k': k,
            'kmedoids': kmedoids,
            'silhouette': silhouette
        }

        if config.operation_mode in [str(OPERATION_MODE.CS), str(OPERATION_MODE.FULL_CS)]:
            new_silhouette, cost = self.heuristic_service.run(k=k,
                                                              graph=graph,
                                                              kmedoids=kmedoids,
                                                              data_props=data_props,
                                                              silhouette=silhouette)
            result['costs'] = cost
            result['silhouette'].update(new_silhouette)

        return result

    @staticmethod
    def _run_kmedoids(data: np.ndarray, k: int) -> dict:
        kmedoids = KMedoids(init='k-medoids++', n_clusters=k, method='pam').fit(data)

        return {
            'labels': kmedoids.labels_,
            'medoids': kmedoids.medoid_indices_,
            'medoids loc': kmedoids.cluster_centers_
        }

    @staticmethod
    def _get_silhouette_value(data: np.ndarray, labels: np.ndarray, centroids: np.ndarray):
        sil_results = {}

        if config.operation_mode in [str(OPERATION_MODE.FULL_CS), str(OPERATION_MODE.FULL_GBAFS)]:
            sil_results['Silhouette'] = get_silhouette_value(X=data, labels=labels, centroids=centroids,
                                                             type='silhouette')
            sil_results['SS'] = get_silhouette_value(X=data, labels=labels, centroids=centroids, type='ss')

        sil_results['MSS'] = get_silhouette_value(X=data, labels=labels, centroids=centroids, type='mss')

        return sil_results

    @staticmethod
    def _log_results(results: list) -> list:
        sorted_results = sorted(results, key=lambda x: x['k'])

        for result in sorted_results:
            k = result['k']
            sil_values = ', '.join(f'({name}) - ({"%.4f" % value})' for name, value in result['silhouette'].items())
            log_service.log(f'[Clustering Service] : Silhouette values for (k={k}) * {sil_values}')

            if config.operation_mode in [str(OPERATION_MODE.CS), str(OPERATION_MODE.FULL_CS)]:
                cost_values = ', '.join(f'({name}) - ({"%.4f" % value})' for name, value in result['costs'].items())
                log_service.log(f'[Clustering Service] : Costs values for (k={k}) * {cost_values}')

        return sorted_results

    @staticmethod
    def _plot_plots(results: list, graph: GraphObject, stage: str, fold_index: int):
        if stage == "Train":
            plot_silhouette(stage=stage,
                            fold_index=fold_index,
                            clustering_results=results)
        if config.operation_mode == str(OPERATION_MODE.FULL_GBAFS):
            plot_clustering(stage=stage,
                            fold_index=fold_index,
                            data=graph.reduced_matrix,
                            clustering_results=results)
            plot_jm_clustering(stage=stage,
                               fold_index=fold_index,
                               data=graph.reduced_matrix,
                               clustering_results=results)
        if config.operation_mode == str(OPERATION_MODE.FULL_CS):
            plot_costs_to_silhouette(stage=stage,
                                     fold_index=fold_index,
                                     clustering_res=results)
