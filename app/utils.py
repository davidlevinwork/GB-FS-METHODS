import copy

from .config import config
from collections import Counter
from .models import OPERATION_MODE


##################################################
# Auxiliary functions for managing train results #
##################################################

def compile_train_results(classification_results: dict, clustering_results: dict) -> dict:
    compiled_results = {
        'clustering': compile_train_clustering_results(results=clustering_results),
        'classification': {}
    }

    if config.operation_mode in [str(OPERATION_MODE.FULL_GBAFS), str(OPERATION_MODE.FULL_CS)]:
        compiled_results['classification'] = compile_train_classification_results(results=classification_results)

    return compiled_results


def compile_train_classification_results(results: dict) -> dict:
    # Init with the results of the first fold
    combined_results = results[0]['Validation']['Results By Classifiers']

    # Sum
    num_of_folds = config.cross_validation.num_splits
    for i in range(1, num_of_folds):
        classifiers = results[i]['Validation']['Results By Classifiers']
        for classifier, classifier_results in classifiers.items():
            combined_results[classifier] = dict(Counter(combined_results[classifier]) + Counter(classifier_results))

    # Divide
    for classifier, classifier_results in combined_results.items():
        combined_results[classifier] = [x / num_of_folds for x in list(combined_results[classifier].values())]

    return combined_results


def compile_train_clustering_results(results: dict) -> list:
    # Init with the results of the first fold
    final_results = init_cluster_results(results=results)

    # Sum
    for i in range(len(results)):
        for j, result in enumerate(results[i]):
            sub_results = result['silhouette']
            for sil_name, sil_value in sub_results.items():
                # Relevant only for heuristic methods: the value will be -1 only when the method didn't succeeded to
                # find any potential solution - so we are doing linear completion based on the previous value
                if sil_value == -1:
                    sil_value = results[i][j - 1]['silhouette'][sil_name]
                    results[i][j]['silhouette'][sil_name] = sil_value
                final_results[j]['silhouette'][sil_name] += sil_value

    # Divide
    for result in final_results:
        k = result['k']
        sub_results = result['silhouette']
        for sil_name, sil_value in sub_results.items():
            if sil_name in ['Silhouette', 'SS', 'MSS']:
                sub_results[sil_name] /= len(results)
            else:
                counter = get_heuristic_method_counter(results=results, method=sil_name, k=k)
                if counter > 0:
                    sub_results[sil_name] /= counter
    return final_results


def init_cluster_results(results):
    clean_results = copy.deepcopy(results[0])

    # Set all silhouette values in combined_results to 0
    for result in clean_results:
        for sil_name in result['silhouette'].keys():
            result['silhouette'][sil_name] = 0
    return clean_results


def get_heuristic_method_counter(results: dict, method: str, k: int) -> int:
    counter = 0
    for result in results.values():
        if result[k - 2]['silhouette'][method] > 0:
            counter += 1
    return counter
