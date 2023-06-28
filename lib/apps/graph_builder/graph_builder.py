import math
import numpy as np
from sklearn.manifold import TSNE

from ...config.config import config
from ...common.helpers import get_distance
from ..io_services.plot_service import plot_tsne
from ...common.models.models import DataObject, GraphData


class GraphBuilder:
    def __init__(self, data: DataObject):
        self.data = data

    def run(self) -> GraphData:
        matrix = self._calculate_separation_matrix()
        reduced_matrix = self._dimensionality_reduction(data=matrix)

        if config.visualization_plots.tsne_plot_enabled:
            plot_tsne(data=reduced_matrix)

        return GraphData(
            matrix=matrix,
            reduced_matrix=reduced_matrix
        )

    def _calculate_separation_matrix(self) -> np.ndarray:
        label_combinations = self._get_label_combinations(self.data.train_data.y.to_numpy())
        separation_matrix = np.zeros((self.data.data_props.n_features,
                                      math.comb(self.data.data_props.n_labels, 2)))

        for i, feature in enumerate(self.data.data_props.features):                 # Iterate over the features
            for j, labels in enumerate(label_combinations):                         # Iterate over each pairs of classes
                separation_matrix[i][j] = get_distance(feature=feature,
                                                       label_1=labels[0],
                                                       label_2=labels[1],
                                                       df=self.data.train_data.x_y)
        return separation_matrix

    @staticmethod
    def _get_label_combinations(labels: np.ndarray) -> list:
        combinations = []
        min_label, max_label = int(np.min(labels)), int(np.max(labels))

        for i_label in range(min_label, max_label + 1):
            for j_label in range(i_label + 1, max_label + 1):
                combinations.append((i_label, j_label))
        return combinations

    @staticmethod
    def _dimensionality_reduction(data: np.ndarray):
        tsne = TSNE(n_components=2,
                    n_iter=config.tsne_algorithm.iterations,
                    perplexity=config.tsne_algorithm.perplexity_value)
        return tsne.fit_transform(X=data)
