import os
import copy
import shutil
from collections import Counter

from .config import config
from .services import log_service
from .models import OPERATION_MODE


EPSILON = 0.01


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

    division_counter = init_division_counter(results=results)
    division_cost_counter = init_cost_division_counter(results=results)

    # Sum
    for i in range(len(results)):
        for j, result in enumerate(results[i]):
            sub_results = result['silhouette']
            for sil_name, sil_value in sub_results.items():
                if sil_value is None:
                    continue
                # Relevant only for heuristic methods: the value will be -1 only when the method didn't succeed to
                # find any potential solution - so we are doing linear completion based on the previous value
                if sil_value == -1:
                    sil_value = results[i][j - 1]['silhouette'][sil_name]
                    results[i][j]['silhouette'][sil_name] = sil_value

                final_results[j]['silhouette'][sil_name] += sil_value
                division_counter[j][sil_name] += 1

    # Divide
    for idx, result in enumerate(final_results):
        sub_results = result['silhouette']
        for sil_name, sil_value in sub_results.items():
            if division_counter[idx][sil_name] > 0:
                sub_results[sil_name] /= division_counter[idx][sil_name]
            else:
                sub_results[sil_name] = None

    for i in range(len(results)):
        for j, result in enumerate(results[i]):
            sub_results = result['costs']
            for cost_name, cost_value in sub_results.items():
                if cost_value is None:
                    continue

                if cost_value == -1:
                    cost_value = results[i][j - 1]['costs'][cost_name]
                    results[i][j]['costs'][cost_name] = cost_value

                final_results[j]['costs'][cost_name] += cost_value
                division_cost_counter[j][cost_name] += 1

    # Divide
    for idx, result in enumerate(final_results):
        sub_results = result['costs']
        for cost_name, cost_value in sub_results.items():
            if division_counter[idx][cost_name] > 0:
                sub_results[cost_name] /= division_counter[idx][cost_name]
            else:
                sub_results[cost_name] = None

    return final_results


def init_cluster_results(results):
    clean_results = copy.deepcopy(results[0])

    # Set all silhouette values in combined_results to 0
    for result in clean_results:
        for sil_name in result['silhouette'].keys():
            result['silhouette'][sil_name] = 0
        for cost_name in result['costs'].keys():
            result['costs'][cost_name] = 0
    return clean_results


def init_division_counter(results):
    counter_list = []

    for result in [x['silhouette'] for x in results[0]]:
        counter_dict = {}
        for sil, sil_value in result.items():
            counter_dict[sil] = 0
        counter_list.append(counter_dict)
    return counter_list


def init_cost_division_counter(results):
    counter_list = []

    for result in [x['costs'] for x in results[0]]:
        counter_dict = {}
        for cost, cost_value in result.items():
            counter_dict[cost] = 0
        counter_list.append(counter_dict)
    return counter_list


def get_legal_knee_value(results: dict):
    knee_value = results['knee_results']['Full MSS']['knee']
    relevant_k_results = sorted([item for item in results['results']['clustering'] if item['k'] <= knee_value],
                                key=lambda x: x['k'], reverse=True)
    for item in relevant_k_results:
        if abs(next((item['silhouette'][key] for key in item['silhouette'] if 'Greedy' in key), 0) -
               item['silhouette']['MSS']) <= EPSILON:
            return item['k']


def clean_up():
    try:
        output_path = os.path.join(os.getcwd(), 'app', 'outputs')
        last_modified_dir = max((d for d in os.listdir(output_path)
                                 if os.path.isdir(os.path.join(output_path, d))),
                                key=lambda d: os.path.getmtime(os.path.join(output_path, d)))

        shutil.copy(os.path.join(output_path, 'Log.txt'), os.path.join(output_path, last_modified_dir))
        os.remove(os.path.join(output_path, 'Log.txt'))
    except Exception as e:
        log_service.log('Critical', f'[Utils] - Failed to save log file. Error: [{e}]')
