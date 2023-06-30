import time
import random
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from ..config import config
from ..services.log_service import log_service
from ..models import (
    DataProps,
    DataObject,
    DataCollection
)


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
            feature_costs=self._generate_feature_costs(features=train.x.columns)
        )

        end_time = time.time()
        log_service.log(f'[Data Service] : Total run time (sec)): [{round(end_time - start_time, 3)}]')

        return DataObject(
            raw_data=df,
            test_data=test,
            train_data=train,
            data_props=data_props,
            normalized_data=normalized_df,
        )

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
        train, test = train_test_split(df, test_size=self.test_size)

        train_obj = DataCollection(x=train.drop(config.data.label_column_name, axis=1),
                                   y=pd.DataFrame(train[config.data.label_column_name]),
                                   x_y=train)

        test_obj = DataCollection(x=test.drop(config.data.label_column_name, axis=1),
                                  y=pd.DataFrame(test[config.data.label_column_name]),
                                  x_y=test)

        return train_obj, test_obj

    @staticmethod
    def _generate_feature_costs(features: pd.DataFrame) -> dict:
        return {feature: random.uniform(0, 2) for feature in features}

    @staticmethod
    def get_fold_split(data: DataObject, train_index: np.ndarray, val_index: np.ndarray) -> tuple:
        train_split = data.train_data.x_y.iloc[train_index]
        train = DataCollection(
            x=train_split.drop(config.data.label_column_name, axis=1),
            y=pd.DataFrame(train_split[config.data.label_column_name]),
            x_y=train_split
        )

        val_split = data.train_data.x_y.iloc[val_index]
        val = DataCollection(
            x=val_split.drop(config.data.label_column_name, axis=1),
            y=pd.DataFrame(val_split[config.data.label_column_name]),
            x_y=val_split
        )

        return train, val
