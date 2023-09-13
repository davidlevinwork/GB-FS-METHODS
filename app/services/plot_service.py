import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.spatial import ConvexHull

from ..config import config
from ..services import log_service
from ..models import OPERATION_MODE

dt = datetime.now()
time_stamp = datetime.timestamp(dt)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'brown', 'black', 'orange', 'yellow', 'navy', 'maroon',
          'olive', 'purple', 'turquoise', 'sienna', 'pink']


def save_plot(plot: plt, stage: str, folder_name: str, file_name: str, fold_index: int):
    try:
        if stage == 'Train':
            full_path = os.path.join(os.path.dirname(__file__), '../', 'outputs',
                                     f'{time_stamp}', f'{stage}', f'Fold #{fold_index}', f'{folder_name}')
        else:
            full_path = os.path.join(os.path.dirname(__file__), '../', 'outputs',
                                     f'{time_stamp}', f'{stage}', f'{folder_name}')
        if not os.path.isdir(full_path):
            os.makedirs(full_path)

        plot.savefig(os.path.join(full_path, file_name))
        plot.close()

    except Exception as e:
        log_service.log('Critical', f'[Plot Service] - Failed to save {file_name} graph. Error: [{e}]')


def plot_tsne(data: np.ndarray, stage: str, fold_index: int):
    try:
        if stage == 'Test':
            return

        plt.clf()
        plt.figure(figsize=(8, 6))

        c = data[:, 0] + data[:, 1]
        plt.scatter(x=data[:, 0], y=data[:, 1], marker='o', c=c, cmap='Wistia')

        plt.xlabel(r'$\lambda_1\psi_1$')
        plt.ylabel(r'$\lambda_2\psi_2$')
        plt.colorbar()
        plt.title(f't-SNE Result')

        save_plot(plot=plt, stage=stage, folder_name='Dimensionality Reduction', file_name='t-SNE',
                  fold_index=fold_index)
    except Exception as e:
        log_service.log('Critical', f'[Plot Service] - Failed to plot t_sne graph. Error: [{e}]')


def plot_silhouette(clustering_results: list, stage: str, fold_index: int):
    try:
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 6))

        for sil_type, color in zip(list(clustering_results[0]['silhouette'].keys()), colors):
            if sil_type not in ['MSS', 'SS', 'Silhouette']:
                continue
            sil_values = [res['silhouette'][sil_type] for res in clustering_results if sil_type in res['silhouette']]
            k_indices = list(range(2, len(sil_values) + 2))
            ax.plot(k_indices, sil_values, label=sil_type, linestyle='--', c=color)

        y_ticks = []
        for i in range(1, 10):
            y_ticks.append(i / 10)

        ax.grid(True, linestyle='-.', color='gray')

        # Set x and y-axis limits to start from 0
        ax.set_ylim(0, 1)
        ax.set_xlim(0, max([res['k'] for res in clustering_results]))

        # Show only the bottom and left ticks
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_yticks(y_ticks)
        ax.set_xlabel('K values')
        ax.set_ylabel('Silhouette value')
        ax.set_title('Silhouette Graph')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, shadow=True, fancybox=True)

        plt.tight_layout()
        save_plot(plot=plt, stage=stage, folder_name='Silhouette', file_name='Silhouette Graph', fold_index=fold_index)
    except AssertionError as e:
        log_service.log('Critical', f'[Plot Service] - Failed to plot silhouette graph. Error: [{e}]')


def plot_clustering(data: np.ndarray, clustering_results: list, stage: str, fold_index: int, extension: str):
    try:
        if stage == 'Test':
            return

        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 6))

        for clustering_result in clustering_results:
            k = clustering_result['k']
            centroids = clustering_result['kmedoids']['medoid_loc']
            labels = clustering_result['kmedoids']['labels']
            u_labels = np.unique(clustering_result['kmedoids']['labels'])

            for label in u_labels:
                color = 'black'
                medoid_point = centroids[label]
                cluster_points = data[labels == label]
                non_medoid_points = cluster_points[~np.all(cluster_points == medoid_point, axis=1)]
                plt.scatter(non_medoid_points[:, 0], non_medoid_points[:, 1], color=color)

                if len(cluster_points) >= 3:
                    hull = ConvexHull(data[labels == label])
                    plt.fill(data[labels == label, 0][hull.vertices], data[labels == label, 1][hull.vertices],
                             color=color, alpha=0.3)

                    for simplex in hull.simplices:
                        plt.plot(data[labels == label, 0][simplex], data[labels == label, 1][simplex], color=color,
                                 alpha=0.5, linewidth=0.8)

                plt.scatter(centroids[label, 0], centroids[label, 1], marker='^', color='red', linewidth=2)

            # Plot labels
            plt.xlabel(r't-SNE$_1$')
            plt.ylabel(r't-SNE$_2$')
            plt.title(f'Clustering Result [K={k}]')

            # Show only the bottom and left ticks
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            save_plot(plot=plt, stage=stage, folder_name=f'Clustering/{extension} Clustering',
                      file_name=f'Clustering for k={k}', fold_index=fold_index)

    except AssertionError as e:
        log_service.log('Critical', f'[Plot Service] - Failed to plot clustering graph. Error: [{e}]')


    except AssertionError as e:
        log_service.log('Critical', f'[Plot Service] - Failed to plot clustering graph. Error: [{e}]')


def plot_jm_clustering(data: np.ndarray, clustering_results: list, stage: str, fold_index: int, extension: str):
    try:
        if stage == 'Test':
            return

        c = data[:, 0] + data[:, 1]
        for clustering_result in clustering_results:
            plt.clf()
            fig, ax = plt.subplots(figsize=(8, 6))

            k = clustering_result['k']
            centroids = clustering_result['kmedoids']['medoid_loc']
            plt.scatter(data[:, 0], data[:, 1], c=c, cmap='Wistia')
            plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', color='black', facecolors='none',
                        linewidth=1.25, label="GB-AFS")

            plt.xlabel(r't-SNE$_1$')
            plt.ylabel(r't-SNE$_2$')

            # Show only the bottom and left ticks
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.legend()
            plt.tight_layout()
            save_plot(plot=plt, stage=stage, folder_name=f'Clustering/{extension} JM Clustering',
                      file_name=f'JM Clustering for k={k}', fold_index=fold_index)

    except AssertionError as e:
        log_service.log('Critical', f'[Plot Service] - Failed to plot jm-clustering graph. Error: [{e}]')


def plot_accuracy_to_silhouette(results: dict, stage: str = 'Test'):
    try:
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 6))

        if config.operation_mode == str(OPERATION_MODE.FULL_CS):
            first_idx, last_idx = results['heuristic_idx']['first_idx'] + 1, results['heuristic_idx']['last_idx'] + 1
            ax.axvspan(first_idx, last_idx, color='gray', alpha=0.25)
            ax.axvspan(last_idx, results['results']['clustering'][-1]['k'], color='wheat', alpha=0.3)

        # Left Y axis (accuracy)
        c_index = 0
        classifier_labels, classifier_handles = [], []
        for classifier, classifier_val in results['results']['classification'].items():
            label = get_classifier_label(str(classifier))
            x_values = [*range(2, len(results['results']['classification'][classifier]) + 2, 1)]
            ax.plot(x_values, classifier_val, linestyle="--", c=colors[c_index])
            classifier_labels.append(label)
            classifier_handles.append(mlines.Line2D([], [], color=colors[c_index], linestyle='--', label=label))
            c_index += 1

        ax.set_xlabel("k values")
        ax.set_ylabel("Accuracy")
        ax.grid(True, linestyle='-.')
        ax.spines['top'].set_visible(False)

        # Right Y axis (silhouette)
        ax2 = ax.twinx()
        sil_labels, sil_handles = [], []
        for sil_type in list(results['results']['clustering'][0]['silhouette'].keys()):
            if sil_type in ['Silhouette', 'SS']:
                continue
            if sil_type == 'MSS':
                sil_values = [res['silhouette'][sil_type] for res in results['results']['clustering']]
            else:
                sil_values = [res['silhouette'][sil_type] for res in results['results']['clustering']][:last_idx]
            k_values = [res['k'] for res in results['results']['clustering']][:len(sil_values)]
            ax2.plot(k_values, sil_values, linestyle="-", c=colors[c_index])
            sil_labels.append(sil_type)
            sil_handles.append(mlines.Line2D([], [], color=colors[c_index], linestyle='-', label=sil_type))
            c_index += 1

        # Creating a separate legend for each axis
        legend1 = ax.legend(classifier_handles, classifier_labels, title="Classifiers", loc='upper left',
                            bbox_to_anchor=(0, 1.15), ncol=2, shadow=True, fancybox=True, fontsize='xx-small')
        # ax.add_artist(legend1)
        legend2 = ax2.legend(sil_handles, sil_labels, title="Silhouette Values", loc='upper right',
                             bbox_to_anchor=(1, 1.15), ncol=2, shadow=True, fancybox=True, fontsize='xx-small')
        ax2.add_artist(legend2)

        # Knee's
        knee_labels, knee_handles = [], []
        for sil_type, knee_value in results['knee_results'].items():
            if knee_value['knee'] is not None:
                ax2.axvline(x=knee_value['knee'], linestyle=':', c=colors[c_index])
                knee_labels.append(f'{sil_type}: {knee_value["knee"]}')
                knee_handles.append(
                    mlines.Line2D([], [], color=colors[c_index], linestyle=':', label=f'Knee: {sil_type}'))
                c_index += 1

        knee_legend = ax2.legend(knee_handles, knee_labels, title="Knee Values", loc='upper center',
                                 bbox_to_anchor=(0.5, 1.15), ncol=2, shadow=True, fancybox=True, fontsize='xx-small')
        ax2.add_artist(knee_legend)

        ax2.set_ylabel("Silhouette")
        ax2.spines['top'].set_visible(False)

        # Compute the desired number of x-ticks and the step size
        n_ticks = 20
        k_values = [int(res['k']) for res in results['results']['clustering']]
        k_range = max(k_values) - min(k_values)
        step = max(1, k_range // n_ticks)
        # Set the x-ticks
        ax.xaxis.set_ticks(np.arange(min(k_values), max(k_values) + 1, step).astype(int))
        ax.tick_params(axis='x', labelsize=7)

        save_plot(plot=plt, stage=stage, folder_name='Accuracy', file_name='Accuracy-Silhouette', fold_index=0)
    except AssertionError as e:
        log_service.log('Critical', f'[Plot Service] - Failed to plot accuracy to silhouette graph. Error: [{e}]')


def plot_costs_to_silhouette(clustering_res: list, stage: str, fold_index: int):
    try:
        if stage != 'Train':
            return
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 6))

        # Left Y axis (costs)
        c_index = 0
        cost_labels, cost_handles = [], []
        for cost_type in list(clustering_res[0]['costs'].keys()):
            k_values = [int(res['k']) for res in clustering_res]
            cost_values = [res['costs'][cost_type] for res in clustering_res]
            ax.plot(k_values, cost_values, linestyle="--", c=colors[c_index])
            cost_labels.append(cost_type)
            cost_handles.append(mlines.Line2D([], [], color=colors[c_index], linestyle='--', label=cost_type))
            c_index += 1

        ax.set_xlabel("k values")
        ax.set_ylabel("Cost")
        ax.grid(True, linestyle='-.')
        ax.spines['top'].set_visible(False)

        # Right Y axis (silhouette)
        ax2 = ax.twinx()
        sil_labels, sil_handles = [], []
        for sil_type in list(clustering_res[0]['silhouette'].keys()):
            if sil_type in ['Silhouette', 'SS']:
                continue
            k_values = [res['k'] for res in clustering_res]
            sil_values = [res['silhouette'][sil_type] for res in clustering_res]
            ax2.plot(k_values, sil_values, linestyle="-", c=colors[c_index])
            sil_labels.append(sil_type)
            sil_handles.append(mlines.Line2D([], [], color=colors[c_index], linestyle='-', label=sil_type))
            c_index += 1

        # Creating a separate legend for each axis
        ax.legend(cost_handles, cost_labels, title="Costs", loc='upper left',
                  bbox_to_anchor=(0, 1.15), ncol=2, shadow=True, fancybox=True, fontsize='xx-small')
        ax2.legend(sil_handles, sil_labels, title="Silhouette Values", loc='upper right',
                   bbox_to_anchor=(1, 1.15), ncol=2, shadow=True, fancybox=True, fontsize='xx-small')

        # Budget
        c_index += 1
        budget = config.constraint_satisfaction.budget
        ax.axhline(y=budget, linestyle='dotted', c='black')
        ax.text(0, budget, f'Budget (={budget})', color='black')

        ax2.set_ylabel("Silhouette Value")
        ax2.spines['top'].set_visible(False)

        # Compute the desired number of x-ticks and the step size
        n_ticks = 10
        k_range = max(k_values) - min(k_values)
        step = max(1, k_range // n_ticks)  # Ensure step is at least 1
        # Set the x-ticks
        ax.xaxis.set_ticks(np.arange(min(k_values), max(k_values) + 1, step).astype(int))

        save_plot(plot=plt, stage=stage, folder_name='Cost', file_name='Cost-Silhouette', fold_index=fold_index)
    except AssertionError as ex:
        log_service.log('Error', f'[Visualization Service] - Failed to plot costs graph. Error: [{ex}]')


def get_classifier_label(classifier):
    if classifier == "DecisionTreeClassifier":
        return "Dec. Tree"
    if classifier == "KNeighborsClassifier":
        return "KNN"
    if classifier == "RandomForestClassifier":
        return "Ran. Forest"
