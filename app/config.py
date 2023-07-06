import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    path: str
    split_ratio: str
    label_column_name: str


class CrossValidationConfig(BaseModel):
    num_splits: int
    allow_shuffle: bool


class TSNEAlgorithmConfig(BaseModel):
    iterations: int
    perplexity_value: int


class ConstraintSatisfaction(BaseModel):
    budget: int


class Config(BaseModel):
    data: DataConfig
    operation_mode: str
    tsne_algorithm: TSNEAlgorithmConfig
    cross_validation: CrossValidationConfig
    constraint_satisfaction: ConstraintSatisfaction


def load_yaml_config(file_path: str) -> Config:
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    return Config(**data)


# Load and parse the YAML configuration file
config = load_yaml_config('config.yaml')
