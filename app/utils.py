from .config import config
from collections import Counter


##################################################
# Auxiliary functions for managing train results #
##################################################

def compile_train_results(classification_results: dict, clustering_results: dict) -> dict:
    compiled_results = {
        'clustering': compile_train_clustering_results(results=clustering_results),
        'classification': {}
    }

    if config.operation_mode == 'full':
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
    combined_results = results[0]

    # Sum
    for i in range(1, len(results)):
        for j, result in enumerate(results[i]):
            sub_results = result['silhouette']
            for sil_name, sil_value in sub_results.items():
                combined_results[j]['silhouette'][sil_name] += sil_value

    # Divide
    for result in combined_results:
        sub_results = result['silhouette']
        for sil_name, sil_value in sub_results.items():
            sub_results[sil_name] /= len(results)

    return combined_results
