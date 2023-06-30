import os
import time
import numpy as np
import concurrent.futures
from sklearn_extra.cluster import KMedoids

from ..config import config
from .silhouette import get_silhouette_value
from ..services.log_service import log_service
from ..models import DataObject, GraphObject
from ..services.plot_service import plot_silhouette, plot_clustering, plot_jm_clustering


class ClusteringService:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, os.cpu_count() + 4)

    def run(self, data: DataObject, graph: GraphObject, k_range: list, stage: str, fold_index: int) -> list:
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [executor.submit(self._run_cluster_evaluation, graph, k) for k in k_range]
            results = [task.result() for task in concurrent.futures.as_completed(tasks)]

        results = self._log_results(results=results)

        if config.visualization_plots.silhouette_plot_enabled:
            plot_silhouette(clustering_results=results, stage=stage, fold_index=fold_index)
        if config.visualization_plots.cluster_plot_enabled:
            plot_clustering(data=graph.reduced_matrix, clustering_results=results, stage=stage, fold_index=fold_index)
        if config.visualization_plots.jm_cluster_plot_enabled:
            plot_jm_clustering(data=graph.reduced_matrix, clustering_results=results, stage=stage, fold_index=fold_index)

        end_time = time.time()
        log_service.log(f'[Clustering Service] : Total run time (sec): [{round(end_time - start_time, 3)}]')

        return results

    def _run_cluster_evaluation(self, graph: GraphObject, k: int) -> dict:
        kmedoids = self._run_kmedoids(data=graph.reduced_matrix, k=k)
        silhouette = self._get_silhouette_value(data=graph.reduced_matrix,
                                                labels=kmedoids['labels'],
                                                centroids=kmedoids['centroids'])
        return {
            'k': k,
            'kmedoids': kmedoids,
            'silhouette': silhouette
        }

    @staticmethod
    def _run_kmedoids(data: np.ndarray, k: int) -> dict:
        kmedoids = KMedoids(init='k-medoids++', n_clusters=k, method='pam').fit(data)

        return {
            'labels': kmedoids.labels_,
            'features': kmedoids.medoid_indices_,
            'centroids': kmedoids.cluster_centers_
        }

    @staticmethod
    def _get_silhouette_value(data: np.ndarray, labels: np.ndarray, centroids: np.ndarray):
        sil_results = {}

        if config.operation_mode == 'full':
            sil_results['Silhouette'] = get_silhouette_value(X=data, labels=labels, centroids=centroids, type='silhouette')
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

        return sorted_results
