import time
import math
import numpy as np
from sklearn.manifold import TSNE
from itertools import combinations

from ..config import config
from ..services import log_service
from .distance import get_distance
from ..services.plot_service import plot_tsne
from ..models import DataProps, GraphObject, DataCollection, OPERATION_MODE


class GraphBuilder:
    def __init__(self, data: DataCollection, data_props: DataProps, fold_index: int):
        self.data = data
        self.data_props = data_props
        self.fold_index = fold_index

    def run(self, stage: str) -> GraphObject:
        matrix = self._calculate_separation_matrix()
        reduced_matrix = self._dimensionality_reduction(data=matrix)

        if config.operation_mode in [str(OPERATION_MODE.FULL_GB_AFS), str(OPERATION_MODE.FULL_GB_BC_FS)]:
            plot_tsne(data=reduced_matrix, stage=stage, fold_index=self.fold_index)

        return GraphObject(
            matrix=matrix,
            reduced_matrix=reduced_matrix
        )

    def _calculate_separation_matrix(self) -> np.ndarray:
        label_combinations = self._get_label_combinations(self.data.y.to_numpy())
        separation_matrix = np.zeros((self.data_props.n_features,
                                      math.comb(self.data_props.n_labels, 2)))

        for i, feature in enumerate(self.data_props.features):                      # Iterate over the features
            for j, labels in enumerate(label_combinations):                         # Iterate over each pairs of classes
                separation_matrix[i][j] = get_distance(feature=feature,
                                                       label_1=labels[0],
                                                       label_2=labels[1],
                                                       df=self.data.x_y)
            log_service.log(f'[Graph Builder] : Computed separation value of feature ({feature}) index ({i+1})')
        return separation_matrix

    @staticmethod
    def _get_label_combinations(labels: np.ndarray) -> list:
        return list(combinations(np.unique(labels), 2))

    @staticmethod
    def _dimensionality_reduction(data: np.ndarray):
        start_time = time.time()

        tsne = TSNE(n_components=2,
                    n_iter=config.tsne_algorithm.iterations,
                    perplexity=config.tsne_algorithm.perplexity_value)
        low_data = tsne.fit_transform(X=data)

        end_time = time.time()
        log_service.log(f'[Graph Builder] : Total run time (sec): [{round(end_time - start_time, 3)}]')
        return low_data
