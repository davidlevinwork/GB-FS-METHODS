import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict
from dataclasses import dataclass


@dataclass
class DataCollection:
    x: pd.DataFrame
    y: pd.DataFrame
    x_y: pd.DataFrame


@dataclass
class DataProps:
    n_labels: int
    labels: np.ndarray
    n_features: int
    features: pd.DataFrame
    feature_costs: Dict[str, float]


@dataclass
class DataObject:
    data_props: DataProps
    test_data: DataCollection
    train_data: DataCollection
    raw_data: pd.DataFrame
    normalized_data: pd.DataFrame


@dataclass
class GraphObject:
    matrix: np.ndarray
    reduced_matrix: np.ndarray


class OPERATION_MODE(Enum):
    GB_AFS = 'GB_AFS'
    FULL_GB_AFS = 'Full-GB_AFS'
    GB_BC_FS = 'GB_BC_FS'
    FULL_GB_BC_FS = 'Full-GB_BC_FS'

    def __str__(self):
        return self.value
