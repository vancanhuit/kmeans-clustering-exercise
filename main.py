import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

""" K-Means Clustering """

# Generate clusters using multivariate normal distribution
means = [[2, 2], [8, 3], [3, 6]]  # Expected centers
covariance_matrix = [[1, 0], [0, 1]]
num_points_per_clusters = 10
num_clusters = 3

cluster0 = np.random.multivariate_normal(
    means[0], 
    covariance_matrix, 
    num_points_per_clusters)
cluster1 = np.random.multivariate_normal(
    means[1], 
    covariance_matrix, 
    num_points_per_clusters)
cluster2 = np.random.multivariate_normal(
    means[2], 
    covariance_matrix, 
    num_points_per_clusters)


points = np.concatenate((cluster0, cluster1, cluster2), axis=0)


def kmeans_display():
    plt.plot(cluster0[:, 0], cluster0[:, 1], '.')
    plt.plot(cluster1[:, 0], cluster1[:, 1], '.')
    plt.plot(cluster2[:, 0], cluster2[:, 1], '.')

    plt.axis('equal')
    plt.plot()
    plt.show()


kmeans_display()
