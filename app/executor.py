import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import config
from .utils import compile_train_results
from .models import DataObject, DataProps
from .services.log_service import log_service
from .clustering.clustering import ClusteringService
from .data_graphing.knee_locator import get_knee
from .data_graphing.graph_builder import GraphBuilder
from .data_graphing.data_processor import DataProcessor
from .services.plot_service import plot_accuracy_to_silhouette
from .classification.classification import ClassificationService


class Executor:
    def __init__(self):
        self.k_fold = KFold(n_splits=config.cross_validation.num_splits,
                            shuffle=config.cross_validation.allow_shuffle)
        self.clustering_service = ClusteringService()
        self.classification_service = ClassificationService()

    def run(self):
        # Prepare the data
        data = DataProcessor().run()
        # STAGE 1 --> Train stage
        knee_results = self._run_train(data=data)
        # STAGE 2 --> Test stage
        final_features = self._run_test(data=data, knee_results=knee_results)

    def _run_train(self, data: DataObject) -> dict:
        log_service.log(f'[Executor] : ******************** Train Stage ********************')
        results = self._get_train_evaluation(data=data)
        knee_results = get_knee(results=results)

        if config.visualization_plots.accuracy_to_silhouette_enabled:
            plot_accuracy_to_silhouette(knee_res=knee_results,
                                        clustering_res=results['clustering'],
                                        classification_res=results['classification'])

        return knee_results

    def _get_train_evaluation(self, data: DataObject):
        clustering_results = {}
        classification_results = {}

        for i, (train_index, val_index) in enumerate(self.k_fold.split(data.train_data.x_y)):
            log_service.log(f'[Executor] : ******************** Fold Number #{i+1} ********************')

            train, validation = DataProcessor.get_fold_split(data=data,
                                                             val_index=val_index,
                                                             train_index=train_index)
            split_data = {
                'train': train,
                'validation': validation
            }
            results = self._run_model(stage="Train",
                                      fold_index=i+1,
                                      data=split_data,
                                      data_props=data.data_props,
                                      k_range=[*range(2, len(data.data_props.features), 1)])

            clustering_results[i] = results['clustering']
            if config.operation_mode == 'full':
                classification_results[i] = results['classification']

        train_results = compile_train_results(clustering_results=clustering_results,
                                              classification_results=classification_results)
        return train_results

    def _run_model(self, data: dict, data_props: DataProps, k_range: list, stage: str, fold_index: int):
        # Calculate separation matrix & Create new (reduced) feature graph
        graph_data = GraphBuilder(data=data['train'],
                                  data_props=data_props,
                                  fold_index=fold_index).run(stage=stage)

        # Execute clustering service (K-Medoid + Silhouette)
        clustering_results = self.clustering_service.run(stage=stage,
                                                         k_range=k_range,
                                                         graph=graph_data,
                                                         data=data['train'],
                                                         fold_index=fold_index)
        # Ignore classification service in 'basic' mode
        if config.operation_mode == 'basic':
            return {'clustering': clustering_results}

        # Execute classification service (Evaluation + Tables)
        classification_results = self.classification_service.run(data=data,
                                                                 stage=stage,
                                                                 k_range=k_range,
                                                                 graph=graph_data,
                                                                 fold_index=fold_index,
                                                                 clustering_res=clustering_results,
                                                                 feature_names=list(data_props.features))

        return {'clustering': clustering_results,
                'classification': classification_results}

    def _run_test(self, data: DataObject, knee_results: dict) -> list:
        log_service.log(f'[Executor] : ******************** Test Stage ********************')

        results = self._run_model(stage="Test",
                                  fold_index=0,
                                  data_props=data.data_props,
                                  data={'train': data.train_data},
                                  k_range=[knee_results['knee']])
        return results['clustering'][0]['kmedoids']['features']
