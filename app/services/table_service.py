import os
import glob
from prettytable import PrettyTable

from .log_service import log_service


def create_table(fold_index: int, mode: str, classification_res: dict):
    train_results = test_results = None

    metrics = ['F1', 'AUC-ovo', 'Accuracy', 'AUC-ovr']

    for metric in metrics:
        if 'train' in classification_res:
            train_results = classification_res['train']['Results By k']
            headers = ['k Value'] + classification_res['train']['Results By k'][0]['Classifiers']
        if 'test' in classification_res:
            test_results = classification_res['test']['Results By K']
            headers = ['k Value'] + classification_res['test']['Results By k'][0]['Classifiers']

        seperator = ['*'] * len(headers)
        table = PrettyTable([header for header in headers])

        if train_results is not None:
            for train_result in train_results:
                row = [train_result['k']] + list(train_result[metric].values())
                table.add_row([col for col in row])

        if train_results is not None and test_results is not None:
            table.add_row([sep for sep in seperator])

        if test_results is not None:
            for test_result in test_results:
                row = [test_result['K']] + list(test_result[metric].values())
                table.add_row([col for col in row])

        save_table(table=table, metric=metric, fold_index=fold_index, mode=mode)


def save_table(table: PrettyTable, metric: str, fold_index: int, mode: str):
    try:
        plots_dir = os.path.join(os.path.dirname(__file__), '..', 'Outputs')
        latest_plot_dir = max(glob.glob(os.path.join(plots_dir, '*/')), key=os.path.getmtime).rsplit('\\', 1)[0]

        if mode == "Train":
            table_file = os.path.join(latest_plot_dir, f'Fold #{fold_index}', 'Metrics', f'{metric} Results.txt')
        else:
            dir = os.path.join(latest_plot_dir, mode)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            table_file = os.path.join(latest_plot_dir, mode, f'{metric} Results.txt')

        with open(table_file, 'w') as w:
            data = table.get_string(title=f"Classification Results - {metric}")
            w.write(data)

    except OSError as e:
        log_service.log('Critical', f'[Table Service] - An error occurred while trying to save the table: {str(e)}')
