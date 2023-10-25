# Graph-Based Feature Selection Methods
Official implementation of two novel graph based feature selection methods: 
1. **GB-AFS:** GB-AFS: ***"Graph-Based Automatic Feature Selection for Multi-Class Classification via Mean Simplified Silhouette"*** [**(link to paper)**](https://arxiv.org/pdf/2309.02272.pdf)
2. **GB-BC-FS:** ***"Graph-Based Feature Selection Method Under Budget Constraints for Multi-Class Classification Problems"*** [**(link to paper)**](https://arxiv.org/pdf/2309.02272.pdf)

## Setup Environment

### 1. Clone the Repo

```bash
git clone https://github.com/davidlevinwork/GB-AFS.git
cd GB-AFS
```

### 2. Install Poetry (if not installed)

- macOS / Linux: 
  ```bash
  curl -sSL https://install.python-poetry.org | bash
  ```
- Windows (PowerShell): 
  ```bash
  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
  ```
- [More Info & Troubleshooting](https://python-poetry.org/docs/#installation)

### 3. Set Up & Activate Virtual Environment
 ```bash
 poetry install
 poetry shell
 ```

---

## Configuration File Explanation
This section provides explanations for each parameter in the configuration file. 
<br /> <br />
Before running the model, you need to fill in the ***config.yaml*** file, which contains important configurations required for the code to run properly. This file should be located in the root folder of the project. Please make sure to set the appropriate values for your specific use case. 

### *mode*
Determines the run mode for the algorithm. There are two options:
- `GB_AFS`: Runs the GB-AFS model only (finds the features' subset)
- `Full-GB_AFS`: Runs the GB-AFS model in proof of concept configuration (including classification, full plots, benchmarks, and results comparison)
- `GB_BC_FS`: Runs the GB-BC-FS model only (finds the features' subset under budget const)
- `Full-GB_BC_FS`: Runs the GB-BC-FS model in proof of concept configuration (including classification, full plots, benchmarks, and results comparison)
### *data*
Dataset related parameters.
- `path`: Relative path to the dataset file
- `split_ratio`: Format "%d-%d" to specify the train-test split ratio
- `label_column_name`: The name of the column containing the data labels
### *cross_validation*
K-fold cross-validation settings.
- `num_splits`: Number of splits for k-fold cross-validation
- `allow_shuffle`: Whether to shuffle the data before splitting it into folds
### *t_sne*
t-SNE algorithm settings.
- `iterations`: Number of iterations for optimization
- `perplexity_value`: The perplexity value must be LOWER than the number of features in the given dataset
### *budget_constraint*
Definitions of budget constraint problems.
- `budget`: The (max) budget defined for solving the problem
- `generate_costs`: Whether to generate costs for features (in cases where there is no information on the costs of the features in the dataset)
- `cost_column_name`: The name of the column containing the feature costs


## Citation
If you find one of the methods to be useful in your own research, please consider citing the following paper:
- Graph-Based Automatic Feature Selection (GB-AFS)
```bib
@article{levin2023graph,
  title={Graph-Based Automatic Feature Selection for Multi-Class Classification via Mean Simplified Silhouette},
  author={Levin, David and Singer, Gonen},
  journal={arXiv preprint arXiv:2309.02272},
  year={2023}
}
```
- Graph-Based Budget-Constrained Feature Selection (GB-BC-FS)
```bib
@article{levin2023graph,
  title={Graph-Based Automatic Feature Selection for Multi-Class Classification via Mean Simplified Silhouette},
  author={Levin, David and Singer, Gonen},
  journal={arXiv preprint arXiv:2309.02272},
  year={2023}
}
```
