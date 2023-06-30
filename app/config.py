import yaml
from pydantic import BaseModel


class VisualizationPlots(BaseModel):
    tsne_plot_enabled: bool
    silhouette_plot_enabled: bool
    cluster_plot_enabled: bool
    jm_cluster_plot_enabled: bool
    accuracy_to_silhouette_enabled: bool


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


class Config(BaseModel):
    operation_mode: str
    data: DataConfig
    cross_validation: CrossValidationConfig
    tsne_algorithm: TSNEAlgorithmConfig
    visualization_plots: VisualizationPlots


def load_yaml_config(file_path: str) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        Config: An instance of the Config model with parsed configuration values.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    return Config(**data)


# Load and parse the YAML configuration file
config = load_yaml_config('config.yaml')