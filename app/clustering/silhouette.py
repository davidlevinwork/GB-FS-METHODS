import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

from ..services.log_service import log_service


def get_silhouette_value(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray, type: str):
    """
    Compute the optimized simplified silhouette score for a set of data points, labels, and centroids.

    Args:
        X (np.ndarray): The feature matrix with shape (n_samples, n_features).
        labels (np.ndarray): The cluster labels for each data point with shape (n_samples,).
        centroids (np.ndarray): The cluster centroids with shape (n_clusters, n_features).
        type (str): The mode for silhouette computation

    Returns:
        float: The mean optimized simplified silhouette score.
    """
    try:
        if type not in ['silhouette', 'ss', 'mss']:
            raise ValueError("Mode must be one of 'silhouette', 'ss', or 'mss'")

        if type == 'silhouette':
            return silhouette_score(X=X, labels=labels)

        a = euclidean_distances(X, centroids[labels]).diagonal()
        b = np.empty_like(a)
        for idx, centroid in enumerate(centroids):
            not_x_centroid = np.delete(centroids, idx, axis=0)
            distances_to_other_centroids = euclidean_distances(X[labels == idx], not_x_centroid)

            if type == 'ss':
                b[labels == idx] = distances_to_other_centroids.min(axis=1)
            elif type == 'mss':
                b[labels == idx] = distances_to_other_centroids.mean(axis=1)

        if type == 'mss':
            mask = a != 0
            a = a[mask]
            b = b[mask]

        sil_values = (b - a) / (np.maximum(a, b))

        if type == 'ss':
            sil_values[sil_values == 1] = 0

        return np.mean(sil_values)
    except Exception as e:
        log_service.log('Critical', f'[Silhouette] - Failed to calculate Silhouette value. Error: [{e}]')
