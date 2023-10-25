import os
import time
import numpy as np
from copy import deepcopy
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

        updated_results = [item['updated'] for item in results]
        original_results = [item['original'] for item in results]

        updated_results = self._log_results(results=updated_results)
        original_results = self._log_results(results=original_results, log=False)

        self._plot_plots(stage=stage, graph=graph, results=updated_results, fold_index=fold_index, extension='Updated')
        self._plot_plots(stage=stage, graph=graph, results=original_results, fold_index=fold_index, extension='Original')

        end_time = time.time()
        log_service.log(f'[Clustering Service] : Total run time (sec): [{round(end_time - start_time, 3)}]')

        return updated_results

    def _run_cluster_evaluation(self, data_props: DataProps, graph: GraphObject, k: int) -> dict:
        kmedoids = self._run_kmedoids(data=graph.reduced_matrix, k=k)
        silhouette = self._get_silhouette_value(data=graph.reduced_matrix,
                                                labels=kmedoids['labels'],
                                                centroids=kmedoids['medoid_loc'])
        result = {
            'k': k,
            'kmedoids': kmedoids,
            'silhouette': silhouette
        }

        original_result = deepcopy(result)

        if config.operation_mode in [str(OPERATION_MODE.GB_BC_FS), str(OPERATION_MODE.FULL_GB_BC_FS)]:
            heuristic_results = self.heuristic_service.run(k=k,
                                                           graph=graph,
                                                           kmedoids=kmedoids,
                                                           data_props=data_props,
                                                           silhouette=silhouette)
            result['costs'] = heuristic_results['cost']
            result['silhouette'].update(heuristic_results['mss'])
            if heuristic_results['is_new_features']:
                result['kmedoids']['labels'] = heuristic_results['new_labels']
                result['kmedoids']['medoids'] = heuristic_results['new_medoids']
                result['kmedoids']['medoid_loc'] = heuristic_results['new_medoids_loc']

        return {
            'original': original_result,
            'updated': result
        }

    @staticmethod
    def _run_kmedoids(data: np.ndarray, k: int) -> dict:
        kmedoids = KMedoids(init='k-medoids++', n_clusters=k, method='pam').fit(data)

        return {
            'labels': kmedoids.labels_,
            'medoids': kmedoids.medoid_indices_,
            'medoid_loc': kmedoids.cluster_centers_
        }

    @staticmethod
    def _get_silhouette_value(data: np.ndarray, labels: np.ndarray, centroids: np.ndarray):
        sil_results = {}

        if config.operation_mode in [str(OPERATION_MODE.FULL_GB_BC_FS), str(OPERATION_MODE.FULL_GB_AFS)]:
            sil_results['Silhouette'] = get_silhouette_value(X=data, labels=labels, centroids=centroids,
                                                             type='silhouette')
            sil_results['SS'] = get_silhouette_value(X=data, labels=labels, centroids=centroids, type='ss')

        sil_results['MSS'] = get_silhouette_value(X=data, labels=labels, centroids=centroids, type='mss')

        return sil_results

    @staticmethod
    def _log_results(results: list, log: bool = True) -> list:
        sorted_results = sorted(results, key=lambda x: x['k'])

        if log:
            for result in sorted_results:
                k = result['k']
                sil_values = ', '.join(f'({name}) - ({("%.4f" % value) if value is not None else "NA"})'
                                       for name, value in result['silhouette'].items())
                log_service.log(f'[Clustering Service] : Silhouette values for (k={k}) * {sil_values}')

                if config.operation_mode in [str(OPERATION_MODE.GB_BC_FS), str(OPERATION_MODE.FULL_GB_BC_FS)]:
                    cost_values = ', '.join(f'({name}) - ({("%.4f" % value) if value is not None else "NA"})'
                                            for name, value in result['costs'].items())
                    log_service.log(f'[Clustering Service] : Costs values for (k={k}) * {cost_values}')

        return sorted_results

    @staticmethod
    def _plot_plots(results: list, graph: GraphObject, stage: str, fold_index: int, extension: str):
        if stage == "Train":
            plot_silhouette(stage=stage,
                            fold_index=fold_index,
                            clustering_results=results)
        if config.operation_mode in (str(OPERATION_MODE.FULL_GB_AFS), str(OPERATION_MODE.FULL_GB_BC_FS)):
            plot_clustering(stage=stage,
                            extension=extension,
                            fold_index=fold_index,
                            data=graph.reduced_matrix,
                            clustering_results=results)
            plot_jm_clustering(stage=stage,
                               extension=extension,
                               fold_index=fold_index,
                               data=graph.reduced_matrix,
                               clustering_results=results)
        if config.operation_mode == str(OPERATION_MODE.FULL_GB_BC_FS) and extension == 'Updated':
            plot_costs_to_silhouette(stage=stage,
                                     fold_index=fold_index,
                                     clustering_res=results)
