import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


def plot_silhouettes(models : list, labels : list, data_proc : pl.DataFrame):
    """
    Plots silhouette scores for a list of clustering models.

    Parameters:
    - models: list of clustering models
    - labels: list of labels corresponding to the models
    - data_proc: processed data (scaled) used for clustering
    """
    assert len(models) == len(labels), "Number of models and number of labels don't match"
    
    for model, label in zip(models, labels):
        fig, ax = plt.subplots(1, 1)
        n_clusters = model.n_clusters
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, len(data_proc) + (n_clusters + 1) * 10])

        cluster_labels = model.fit_predict(data_proc)

        sil = silhouette_samples(data_proc, cluster_labels)
        avg_sil = sil.mean()

        print(f'Avg. Silhouette for {label}: {avg_sil}')

        y_lower = 10

        for i in range(n_clusters):
            ith_cluster_silhouette_values = sil[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10 

        ax.set_title(f"Silhouette plot \n {label}")
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        ax.axvline(x=avg_sil, color="red", linestyle="--")

        ax.set_yticks([])

        plt.show()
        plt.close()