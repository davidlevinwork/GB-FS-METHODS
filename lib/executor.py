import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from lib.apps.io_services.log_service import log_service
from .apps.clustering.clustering import ClusteringService
from .apps.data_graphing.graph_builder import GraphBuilder
from .apps.data_graphing.data_processor import DataProcessor
from lib.common.models.models import DataObject, DataProps
from .apps.classification.classification import ClassificationService

from .config.config import config


class Executor:
    def __init__(self):
        self.k_fold = KFold(n_splits=config.cross_validation.num_splits, shuffle=config.cross_validation.allow_shuffle)
        self.clustering_service = ClusteringService()
        self.classification_service = ClassificationService()

    def run(self):
        # Prepare the data
        data = DataProcessor().run()
        # train
        self._train(data=data)

    def _train(self, data: DataObject):
        clustering_results = {}
        classification_results = {}

        for i, (train_index, val_index) in enumerate(self.k_fold.split(data.train_data.x_y)):
            log_service.log(f'[Executor] : ******************** Fold Number #{i + 1} ********************')

            train, validation = DataProcessor.get_fold_split(data=data,
                                                             val_index=val_index,
                                                             train_index=train_index)
            split_data = {
                'train': train,
                'validation': validation
            }
            results = self._run_model(fold_index=i + 1,
                                      data=split_data,
                                      data_props=data.data_props,
                                      k_range=[*range(2, len(data.data_props.features), 1)])

            clustering_results[i] = results['clustering']
            if config.operation_mode == 'full':
                classification_results[i] = results['classification']

        x = 5

    def _run_model(self, data: dict, data_props: DataProps, k_range: list, fold_index: int):
        # Calculate separation matrix & Create new (reduced) feature graph
        graph_data = GraphBuilder(data=data['train'],
                                  data_props=data_props,
                                  fold_index=fold_index).run()

        # Execute clustering service (K-Medoid + Silhouette)
        clustering_results = self.clustering_service.run(k_range=k_range,
                                                         graph=graph_data,
                                                         data=data['train'],
                                                         fold_index=fold_index)
        # Pass classification service in 'basic' mode
        if config.operation_mode == 'basic':
            return {'clustering': clustering_results}

        classification_results = self.classification_service.run(data=data,
                                                                 mode="Train",
                                                                 k_range=k_range,
                                                                 graph=graph_data,
                                                                 clustering_res=clustering_results,
                                                                 feature_names=list(data_props.features))

        return {'clustering': clustering_results,
                'classification': classification_results}

    def _full_train(self):
        # stage 2
        pass
