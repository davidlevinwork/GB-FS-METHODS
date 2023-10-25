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


class BudgetConstraint(BaseModel):
    budget: float
    generate_costs: bool


class Config(BaseModel):
    data: DataConfig
    operation_mode: str
    tsne_algorithm: TSNEAlgorithmConfig
    cross_validation: CrossValidationConfig
    budget_constraint: BudgetConstraint


def load_yaml_config(file_path: str) -> Config:
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    return Config(**data)


# Load and parse the YAML configuration file
config = load_yaml_config('config.yaml')
