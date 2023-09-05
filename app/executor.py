from sklearn.model_selection import KFold

from .config import config
from .services import log_service
from .clustering import ClusteringService
from .services.table_service import create_table
from .data_graphing.knee_locator import get_knees
from .classification import ClassificationService
from .data_graphing.graph_builder import GraphBuilder
from .data_graphing.data_processor import DataProcessor
from .models import DataObject, DataProps, OPERATION_MODE
from .services.plot_service import plot_accuracy_to_silhouette
from .utils import compile_train_results, get_legal_knee_value
from .classification.benchmarking import select_k_best_features


class Executor:
    def __init__(self):
        self.k_fold = KFold(n_splits=config.cross_validation.num_splits,
                            shuffle=config.cross_validation.allow_shuffle)
        self.clustering_service = ClusteringService()
        self.classification_service = ClassificationService()

    def run(self):
        # Prepare the data
        data = DataProcessor().run()
        # STAGE 1 --> Train stage
        knee_results = self._run_train(data=data)
        # STAGE 2 --> Test stage
        final_features = self._run_test(data=data, knee_results=knee_results)

        if config.operation_mode in [str(OPERATION_MODE.FULL_GBAFS), str(OPERATION_MODE.FULL_CS)]:
            # Stage 3 --> Evaluate test stage (selected features)
            self._run_test_evaluation(data=data, features=final_features)
            # Stage 4 --> Benchmark evaluation
            self._run_benchmark_evaluation(data=data, knee_results=knee_results)

    def _run_train(self, data: DataObject) -> dict:
        results = self._get_train_evaluation(data=data)
        final_results = get_knees(results=results)

        if config.operation_mode == str(OPERATION_MODE.FULL_CS):
            old_knee = final_results['knee_results']['Full MSS']['knee']
            new_knee = get_legal_knee_value(results=final_results)

            if new_knee != old_knee:
                final_results['knee_results']['Heuristic MSS'] = {}
                final_results['knee_results']['Heuristic MSS']['knee'] = get_legal_knee_value(results=final_results)

            log_service.log(f"[Executor] : 'Original' knee based on MSS graph = [{old_knee}]. "
                            f"'Updated' knee based on Heuristic MSS graph = [{new_knee}]")

        if config.operation_mode in [str(OPERATION_MODE.FULL_GBAFS), str(OPERATION_MODE.FULL_CS)]:
            plot_accuracy_to_silhouette(results=final_results)
        return final_results['knee_results']

    def _run_test(self, data: DataObject, knee_results: dict) -> list:
        log_service.log(f'[Executor] : ******************** Test Stage ********************')

        results = self._run_model(stage="Test",
                                  fold_index=0,
                                  data_props=data.data_props,
                                  data={'train': data.train_data},
                                  k_range=[value['knee'] for key, value in knee_results.items() if key == 'Full MSS'])

        final_features = results['clustering'][0]['kmedoids']['medoids']
        log_service.log(f'[Executor] : ===> Final k=[{len(final_features)}] features selected are: '
                        f'[{", ".join(map(str, final_features))}] <===')
        return final_features

    def _get_train_evaluation(self, data: DataObject):
        clustering_results = {}
        classification_results = {}

        for i, (train_index, val_index) in enumerate(self.k_fold.split(data.train_data.x_y)):
            log_service.log(f'[Executor] : ******************** Fold Number #{i + 1} ********************')

            train, validation = DataProcessor.get_fold_split(data=data,
                                                             val_index=val_index,
                                                             train_index=train_index)
            split_data = {
                'train': train,
                'validation': validation
            }
            results = self._run_model(stage="Train",
                                      fold_index=i + 1,
                                      data=split_data,
                                      data_props=data.data_props,
                                      k_range=[*range(2, len(data.data_props.features), 1)])

            clustering_results[i] = results['clustering']
            if config.operation_mode in [str(OPERATION_MODE.FULL_GBAFS), str(OPERATION_MODE.FULL_CS)]:
                classification_results[i] = results['classification']

        train_results = compile_train_results(clustering_results=clustering_results,
                                              classification_results=classification_results)
        return train_results

    def _run_model(self, data: dict, data_props: DataProps, k_range: list, stage: str, fold_index: int):
        # Calculate separation matrix & Create new (reduced) feature graph
        graph_data = GraphBuilder(data=data['train'],
                                  data_props=data_props,
                                  fold_index=fold_index).run(stage=stage)

        # Execute clustering service (K-Medoid + Silhouette)
        clustering_results = self.clustering_service.run(stage=stage,
                                                         k_range=k_range,
                                                         graph=graph_data,
                                                         data_props=data_props,
                                                         fold_index=fold_index)
        # Ignore classification service in 'basic' modes
        if config.operation_mode in [str(OPERATION_MODE.GBAFS), str(OPERATION_MODE.CS)]:
            return {'clustering': clustering_results}

        # Execute classification service (Evaluation + Tables)
        classification_results = self.classification_service.run(data=data,
                                                                 stage=stage,
                                                                 k_range=k_range,
                                                                 graph=graph_data,
                                                                 fold_index=fold_index,
                                                                 clustering_res=clustering_results,
                                                                 feature_names=list(data_props.features))

        return {'clustering': clustering_results,
                'classification': classification_results}

    def _run_test_evaluation(self, data: DataObject, features: list):
        log_service.log(f'[Executor] : ********************* Test Evaluation *********************')

        # Execute classification service (evaluation only)
        new_X = data.test_data.x.iloc[:, features]
        classification_res = self.classification_service.evaluate(X=new_X,
                                                                  y=data.test_data.y,
                                                                  mode="Test",
                                                                  k=len(features))
        final_results = self.classification_service.sort_results([classification_res])
        create_table(mode="Test",
                     fold_index=0,
                     classification_res={"Test": final_results})

    def _run_benchmark_evaluation(self, data: DataObject, knee_results: dict):
        log_service.log(f'[Executor] : ********************* Benchmark Evaluations *********************')

        classifications_res = []
        algorithms = ["Relief", "Fisher", "CFS", "MRMR", "Random"]
        k = knee_results['Heuristic MSS']['knee'] if 'Heuristic MSS' in knee_results \
            else knee_results['Full MSS']['knee']

        for algo in algorithms:
            new_k, new_X = select_k_best_features(k=k,
                                                  data=data,
                                                  algorithm=algo)
            classification_res = self.classification_service.evaluate(k=new_k,
                                                                      X=new_X,
                                                                      y=data.test_data.y,
                                                                      mode="Test")
            classifications_res.append(classification_res)

        final_results = self.classification_service.sort_results(classifications_res)

        # Create results table
        create_table(fold_index=0,
                     mode="Benchmarks",
                     algorithms=algorithms,
                     classification_res={"Test": final_results})
