import time
import random

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from ..config import config
from ..services import log_service
from ..models import DataProps, DataObject, DataCollection


class DataProcessor:
    def __init__(self):
        self.train_size, self.test_size = [int(size) / 100 for size in config.data.split_ratio.split("-")]

    def run(self) -> DataObject:
        start_time = time.time()

        df = self._load_data()
        normalized_df = self._normalize_data(df=df)

        train, test = self._train_test_split(df=normalized_df)
        data_props = DataProps(
            labels=train.y[config.data.label_column_name].unique(),
            n_labels=int(train.y.nunique()),
            features=train.x.columns,
            n_features=len(train.x.columns),
            feature_costs=self._set_feature_costs(x=train.x)
        )

        end_time = time.time()
        log_service.log(f'[Data Service] : Total run time (sec): [{round(end_time - start_time, 3)}]')

        return DataObject(
            raw_data=df,
            test_data=test,
            train_data=train,
            data_props=data_props,
            normalized_data=normalized_df,
        )

    @staticmethod
    def get_fold_split(data: DataObject, train_index: np.ndarray, val_index: np.ndarray) -> tuple:
        train_data = data.train_data.x_y.iloc[train_index]
        train = DataCollection(
            x=train_data.drop(config.data.label_column_name, axis=1),
            y=pd.DataFrame(train_data[config.data.label_column_name]),
            x_y=train_data
        )

        val_data = data.train_data.x_y.iloc[val_index]
        val = DataCollection(
            x=val_data.drop(config.data.label_column_name, axis=1),
            y=pd.DataFrame(val_data[config.data.label_column_name]),
            x_y=val_data
        )
        return train, val

    @staticmethod
    def _load_data() -> pd.DataFrame:
        return pd.read_csv(config.data.path)

    @staticmethod
    def _normalize_data(df: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler()
        cols_to_normalize = df.columns.difference([config.data.label_column_name])
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        return df

    def _train_test_split(self, df: pd.DataFrame) -> Tuple[DataCollection, DataCollection]:
        train_data, test_data = train_test_split(df, test_size=self.test_size)

        train = DataCollection(x=train_data.drop(config.data.label_column_name, axis=1),
                               y=pd.DataFrame(train_data[config.data.label_column_name]),
                               x_y=train_data)

        test = DataCollection(x=test_data.drop(config.data.label_column_name, axis=1),
                              y=pd.DataFrame(test_data[config.data.label_column_name]),
                              x_y=test_data)

        return train, test

    @staticmethod
    def _set_feature_costs(x: pd.DataFrame) -> dict:
        epsilon = 1e-10
        features = x.columns
        col_name = config.budget_constraint.cost_column_name

        if not config.budget_constraint.generate_costs:
            if col_name is not None and col_name in features:
                costs = x[col_name].fillna(epsilon).to_list()                       # Ensure no missing values
                return {feature: cost for feature, cost in zip(features, costs)}
        return {feature: random.uniform(0, 2) for feature in features}
