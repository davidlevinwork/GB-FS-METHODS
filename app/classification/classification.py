import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn import tree
import concurrent.futures
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ..services import log_service
from ..models import GraphObject, DataCollection
from ..services.table_service import create_table

warnings.filterwarnings('ignore')

NUMBER_OF_TEST_EPOCHS = 1
NUMBER_OF_TRAIN_EPOCHS = 10


class ClassificationService:
    def __init__(self, max_workers: int = 1):
        self.classifiers = [
            KNeighborsClassifier(),
            RandomForestClassifier(),
            tree.DecisionTreeClassifier()
        ]
        self.cv = KFold(n_splits=10, shuffle=True)
        self.max_workers = max_workers or min(32, os.cpu_count() + 4)

    def run(self, stage: str, data: dict, graph: GraphObject, clustering_res: list, feature_names: list, k_range: list,
            fold_index: int) -> dict:
        if stage not in ['Train', 'Test']:
            raise ValueError("Invalid mode. Allowed values are 'Train' and 'Full Train'.")

        start_time = time.time()

        if stage == 'Train':
            train_results = self._get_classification_evaluation(mode='Train',
                                                                graph=graph,
                                                                k_range=k_range,
                                                                data=data['train'],
                                                                feature_names=feature_names,
                                                                clustering_results=clustering_res)
            val_results = self._get_classification_evaluation(mode='Validation',
                                                              graph=graph,
                                                              k_range=k_range,
                                                              data=data['validation'],
                                                              feature_names=feature_names,
                                                              clustering_results=clustering_res)
            results = {
                "Train": train_results,
                'Validation': val_results
            }
        else:
            test_results = self._get_classification_evaluation(mode='Test',
                                                               graph=graph,
                                                               k_range=k_range,
                                                               data=data['train'],
                                                               feature_names=feature_names,
                                                               clustering_results=clustering_res)
            results = {
                "Train": test_results,
            }
        create_table(mode=stage,
                     fold_index=fold_index,
                     classification_res=results)

        end_time = time.time()
        log_service.log(f'[Classification Service] : Total run time (sec): [{round(end_time - start_time, 3)}]')
        return results

    def _get_classification_evaluation(self, data: DataCollection, graph: GraphObject, feature_names: list, mode: str,
                                       clustering_results: list, k_range: list) -> dict:
        results = []
        if mode in ['Train', 'Validation']:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                tasks = [executor.submit(self._execute_classification, data, graph,
                                         clustering_results[k - 2], k, feature_names, mode) for k in k_range]
                results = [task.result() for task in concurrent.futures.as_completed(tasks)]
        else:
            # The serial process is more effective due to the smaller k_range (just the knee value)
            for k in k_range:
                k_clustering = [x for x in clustering_results if x['k'] == k][0]
                results.append(self._execute_classification(data, graph, k_clustering, k, feature_names, mode))

        results = self.sort_results(results=results)
        return results

    def _execute_classification(self, data: DataCollection, graph: GraphObject, clustering_results: dict, k: int,
                                feature_names: list, mode: str) -> dict:
        new_X = self._prepare_data(X=data.x,
                                   feature_names=feature_names,
                                   reduced_matrix=graph.reduced_matrix,
                                   centroids=clustering_results['kmedoids']['medoids loc'])
        evaluation = self.evaluate(new_X, data.y, k, mode)
        return evaluation

    @staticmethod
    def _prepare_data(X: pd.DataFrame, reduced_matrix: np.ndarray, centroids: np.ndarray, feature_names: list):
        feature_indexes = [i for i in range(len(reduced_matrix)) if reduced_matrix[i] in centroids]
        selected_feature_names = [feature_names[i] for i in feature_indexes]
        new_X = X[X.columns.intersection(selected_feature_names)]

        return new_X

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame, k: int, mode: str) -> dict:
        classifier_strings = []
        classifier_to_accuracy = {}
        classifier_to_f1 = {}
        classifier_to_auc_ovo = {}
        classifier_to_auc_ovr = {}

        lb = LabelBinarizer()
        y_binary = lb.fit_transform(y)                                          # Convert true labels to binary format
        for classifier in self.classifiers:
            accuracy_list, f1_list, auc_ovo_list, auc_ovr_list = [], [], [], []
            classifier_str = str(classifier).replace('(', '').replace(')', '')
            classifier_strings.append(classifier_str)

            epochs = NUMBER_OF_TRAIN_EPOCHS if mode == 'Train' else NUMBER_OF_TEST_EPOCHS
            for _ in range(epochs):
                cv_predictions = cross_val_predict(classifier, X, np.ravel(y), cv=self.cv)
                cv_predictions_proba = cross_val_predict(classifier, X, np.ravel(y), cv=self.cv, method='predict_proba')
                accuracy = accuracy_score(y, cv_predictions)
                f1 = f1_score(y, cv_predictions, average='weighted')
                auc_ovo = roc_auc_score(y_binary, cv_predictions_proba, multi_class='ovo', average='weighted')
                auc_ovr = roc_auc_score(y_binary, cv_predictions_proba, multi_class='ovr', average='weighted')

                f1_list.append(f1)
                accuracy_list.append(accuracy)
                auc_ovo_list.append(auc_ovo)
                auc_ovr_list.append(auc_ovr)

            classifier_to_f1[classifier_str] = np.mean(f1_list)
            classifier_to_accuracy[classifier_str] = np.mean(accuracy_list)
            classifier_to_auc_ovo[classifier_str] = np.mean(auc_ovo_list)
            classifier_to_auc_ovr[classifier_str] = np.mean(auc_ovr_list)

        return {
            'k': k,
            'Mode': mode,
            'Classifiers': classifier_strings,
            'F1': classifier_to_f1,
            'AUC-ovo': classifier_to_auc_ovo,
            'Accuracy': classifier_to_accuracy,
            'AUC-ovr': classifier_to_auc_ovr
        }

    def sort_results(self, results: list) -> dict:
        arranged_results = {
            'Results By k': sorted(results, key=lambda x: x['k']),
            'Results By Classifiers': self._arrange_results_by_classifier(results),
            'Results By F1': sorted(results, key=lambda x: sum(x['F1'].values()) / len(x['F1']), reverse=True),
            'Results By AUC-ovo': sorted(results, key=lambda x: sum(x['AUC-ovo'].values()) / len(x['AUC-ovo']),
                                         reverse=True),
            'Results By AUC-ovr': sorted(results, key=lambda x: sum(x['AUC-ovr'].values()) / len(x['AUC-ovr']),
                                         reverse=True),
            'Results By Accuracy': sorted(results, key=lambda x: sum(x['Accuracy'].values()) / len(x['Accuracy']),
                                          reverse=True)
        }
        return arranged_results

    @staticmethod
    def _arrange_results_by_classifier(results: list) -> dict:
        arranged_results = {}
        classifiers = list(results[0]['Classifiers'])

        for classifier in classifiers:
            classifier_results = {}
            for result in results:
                K = result['k']
                classification_results = result['Accuracy']
                for c, res in classification_results.items():
                    if c == classifier:
                        classifier_results[K] = res
            arranged_results[classifier] = dict(sorted(classifier_results.items(), key=lambda item: item[0]))

        return arranged_results
