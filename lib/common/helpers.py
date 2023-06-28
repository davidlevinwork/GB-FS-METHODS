import numpy as np
import pandas as pd
from ..common.config.config import config


def get_distance(df: pd.DataFrame, feature: str, label_1: str, label_2: str) -> float:
    x_1 = np.array(df.loc[df[config.data.label_column_name] == label_1][feature])
    x_2 = np.array(df.loc[df[config.data.label_column_name] == label_2][feature])
    return jm_distance(x_1, x_2)


def jm_distance(p: np.ndarray, q: np.ndarray):
    b = bhattacharyya_distance(p, q)
    jm = 2 * (1 - np.exp(-b))
    return jm


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray):
    mean_p, mean_q = p.mean(), q.mean()
    std_p = p.std() if p.std() != 0 else 0.00000000001
    std_q = q.std() if q.std() != 0 else 0.00000000001

    var_p, var_q = std_p ** 2, std_q ** 2
    b = (1 / 8) * ((mean_p - mean_q) ** 2) * (2 / (var_p + var_q)) + \
        0.5 * np.log((var_p + var_q) / (2 * (std_p * std_q)))
    return b
