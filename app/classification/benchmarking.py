import time
import pandas as pd
from random import sample

from ..services.log_service import log_service
from skfeature.function.statistical_based.CFS import cfs
from skfeature.function.similarity_based.reliefF import reliefF
from skfeature.function.information_theoretical_based.MRMR import mrmr
from skfeature.function.similarity_based.fisher_score import fisher_score


def select_k_best_features(X: pd.DataFrame, y: pd.DataFrame, k: int, algorithm: str):
    """
    Select k best features from the dataset using the specified algorithm.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target vector.
        k (int): Number of features to select.
        algorithm (str): Algorithm to use for feature selection.

    Returns:
        pd.DataFrame: Dataframe with selected features.
    """
    start_time = time.time()

    y = y.to_numpy().reshape(y.shape[0])

    if algorithm == "Relief":
        score = reliefF(X.to_numpy(), y)
        selected_features = X.columns[score.argsort()[-k:]].tolist()
    elif algorithm == "Fisher":
        score = fisher_score(X.to_numpy(), y)
        selected_features = X.columns[score.argsort()[-k:]].tolist()
    elif algorithm == "CFS":
        score = cfs(X.to_numpy(), y)
        selected_features = X.columns[score.argsort()[-k:]].tolist()
    elif algorithm == "MRMR":
        score = mrmr(X.to_numpy(), y, k)
        selected_features = X.columns[score].tolist()
    elif algorithm == "Random":
        selected_features = sample(X.columns.tolist(), k)
    else:
        raise ValueError("Invalid algorithm name")

    end_time = time.time()
    log_service.log(f'[Classification Service] - [Benchmarking]: Total run time (sec) for [{algorithm}]:'
                    f' [{round(end_time - start_time, 3)}]')

    return X[selected_features]
