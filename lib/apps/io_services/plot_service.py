import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


dt = datetime.now()
time_stamp = datetime.timestamp(dt)


def save_plot(plot: plt, file_name: str):
    try:
        full_path = os.path.join(os.path.dirname(__file__), '../../', 'Outputs')

        if not os.path.isdir(full_path):
            os.makedirs(full_path)

        plot.savefig(os.path.join(full_path, file_name))

    except Exception as e:
        print(f"An error occurred while trying to save the plot: {str(e)}")


def plot_tsne(data: np.ndarray):
    try:
        plt.clf()
        plt.figure(figsize=(10, 8))

        c = data[:, 0] + data[:, 1]
        plt.scatter(x=data[:, 0], y=data[:, 1], marker='o', c=c, cmap='Wistia')

        plt.xlabel(r'$\lambda_1\psi_1$')
        plt.ylabel(r'$\lambda_2\psi_2$')
        plt.colorbar()
        plt.title(f't-SNE Result')

        save_plot(plot=plt, file_name='t-SNE')
    except Exception as ex:
        pass
