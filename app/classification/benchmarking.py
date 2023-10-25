import time
import numpy as np
import pandas as pd
from copy import deepcopy
from random import sample

from ..config import config
from ..services import log_service
from ..models import OPERATION_MODE, DataObject
from skfeature.function.statistical_based.CFS import cfs
from skfeature.function.similarity_based.reliefF import reliefF
from skfeature.function.information_theoretical_based.MRMR import mrmr
from skfeature.function.similarity_based.fisher_score import fisher_score


def select_k_best_features(data: DataObject, k: int, algorithm: str):
    k_features = get_k_features(X=data.test_data.x, y=data.test_data.y, k=k, algorithm=algorithm)
    if config.operation_mode == str(OPERATION_MODE.FULL_GB_AFS):
        return k, k_features

    k_features_copy = deepcopy(k_features)
    while k_features_copy.columns.tolist():
        total_cost = sum_feature_costs(features=list(k_features_copy), data=data)
        if total_cost <= config.constraint_satisfaction.budget:
            return len(k_features_copy.columns), k_features_copy
        else:
            k_features_copy.drop(k_features_copy.columns[-1], axis=1, inplace=True)
    return k, k_features


def get_k_features(X: pd.DataFrame, y: pd.DataFrame, k: int, algorithm: str):
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
        selected_features = X.columns[score.argsort()[-k:][np.argsort(score[score.argsort()[-k:]])[::-1]]].tolist()
    elif algorithm == "Fisher":
        score = fisher_score(X.to_numpy(), y)
        selected_features = X.columns[score.argsort()[-k:][np.argsort(score[score.argsort()[-k:]])[::-1]]].tolist()
    elif algorithm == "CFS":
        score = cfs(X.to_numpy(), y)
        selected_features = X.columns[score.argsort()[-k:][np.argsort(score[score.argsort()[-k:]])[::-1]]].tolist()
    elif algorithm == "MRMR":
        score = mrmr(X.to_numpy(), y, k)
        selected_features = X.columns[score.argsort()[-k:][np.argsort(score[score.argsort()[-k:]])[::-1]]].tolist()
    elif algorithm == "Random":
        selected_features = sample(X.columns.tolist(), k)
    else:
        raise ValueError("Invalid algorithm name")

    end_time = time.time()
    log_service.log(f'[Classification Service] - [Benchmarking]: Total run time (sec) for {algorithm} method:'
                    f' [{round(end_time - start_time, 3)}]')

    return X[selected_features]


def sum_feature_costs(features: list, data: DataObject):
    return sum(data.data_props.feature_costs[feature_name] for feature_name in list(features))
