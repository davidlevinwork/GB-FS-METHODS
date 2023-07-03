import os
import glob
from prettytable import PrettyTable

from .log_service import log_service


def create_table(fold_index: int, mode: str, classification_res: dict, algorithms: list = None):
    train_results = test_results = None

    metrics = ['F1', 'AUC-ovo', 'Accuracy', 'AUC-ovr']

    for metric in metrics:
        if 'Train' in classification_res:
            train_results = classification_res['Train']['Results By k']
            headers = ['k Value'] + classification_res['Train']['Results By k'][0]['Classifiers']
        if 'Test' in classification_res:
            test_results = classification_res['Test']['Results By k']
            if algorithms is None:
                headers = ['k Value'] + classification_res['Test']['Results By k'][0]['Classifiers']
            else:
                headers = ['k Value'] + ['Benchmark'] + classification_res['Test']['Results By k'][0]['Classifiers']

        seperator = ['*'] * len(headers)
        table = PrettyTable([header for header in headers])

        if train_results is not None:
            for train_result in train_results:
                row = [train_result['k']] + list(train_result[metric].values())
                table.add_row([col for col in row])

        if train_results is not None and test_results is not None:
            table.add_row([sep for sep in seperator])

        if test_results is not None:
            if algorithms is None:
                for test_result in test_results:
                    row = [test_result['k']] + list(test_result[metric].values())
                    table.add_row([col for col in row])
            else:
                for i, test_result in enumerate(test_results):
                    row = [test_result['k']] + [algorithms[i]] + list(test_result[metric].values())
                    table.add_row([col for col in row])

        save_table(table=table, metric=metric, fold_index=fold_index, mode=mode)


def save_table(table: PrettyTable, metric: str, fold_index: int, mode: str):
    try:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
        latest_output_dir = max(glob.glob(os.path.join(output_dir, '*/')), key=os.path.getmtime).rsplit('\\', 1)[0]

        if mode == "Train":
            tables_dir = os.path.join(latest_output_dir, mode, f'Fold #{fold_index}', 'Metrics')
            if not os.path.isdir(tables_dir):
                os.makedirs(tables_dir)
            table_file = os.path.join(tables_dir, f'{metric} Results.txt')
        else:
            dir = os.path.join(latest_output_dir, mode)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            table_file = os.path.join(latest_output_dir, mode, f'{metric} Results.txt')

        with open(table_file, 'w') as w:
            data = table.get_string(title=f"Classification Results - {metric}")
            w.write(data)

    except OSError as e:
        log_service.log('Critical', f'[Table Service] - An error occurred while trying to save the table: {str(e)}')
