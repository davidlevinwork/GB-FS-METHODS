import random
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from ...common.config.config import config
from ...common.models.models import (
    DataProps,
    DataObject,
    DataCollection
)


class DataProcessor:
    def __init__(self):
        self.train_size, self.test_size = [int(size) / 100 for size in config.data.split_ratio.split("-")]

    def run(self) -> DataObject:
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
