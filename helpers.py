import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

np.random.seed(2017)


def generate_data_points():
    """ To generate data points for testing algorithm, we use multivariate
    normal distribution """
    means = [[1, 2], [7, 3], [4, 9]]  # 3 expected centers for 3 clusters
    covariance_matrix = [[1, 0], [0, 1]]
    num_points_per_clusters = 500
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
    original_labels = np.asarray([0] * num_points_per_clusters +
                                 [1] * num_points_per_clusters +
                                 [2] * num_points_per_clusters)

    return (num_clusters, original_labels, points)


def display(points, labels):
    """ Display data points using matplotlib library """

    # Get each cluster in data points based on its labels
    cluster0 = points[labels == 0, :]
    cluster1 = points[labels == 1, :]
    cluster2 = points[labels == 2, :]

    # Plot each cluster
    plt.plot(cluster0[:, 0], cluster0[:, 1], '.')
    plt.plot(cluster1[:, 0], cluster1[:, 1], '.')
    plt.plot(cluster2[:, 0], cluster2[:, 1], '.')

    plt.axis('equal')
    plt.plot()
    plt.show()


def init_centers(points, num_clusters):
    # randomly pick k rows of points as initial centers
    return points[np.random.choice(points.shape[0],
                                   num_clusters, replace=False)]


def assign_labels(points, centers):
    # calculate distance from each point to centers
    distances = distance.cdist(points, centers, 'euclidean')
    # return index of the closest center
    return np.argmin(distances, axis=1)


def update_centers(points, labels, num_clusters):
    centers = np.zeros((num_clusters, points.shape[1]))
    for k in range(num_clusters):
        cluster = points[labels == k, :]
        centers[k, :] = np.mean(cluster, axis=0)

    return centers


def has_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) ==
            set([tuple(a) for a in new_centers]))


def kmeans(points, num_clusters):
    centers = init_centers(points, num_clusters)
    labels = []
    it = 0
    converged = False

    while not converged:
        new_labels = assign_labels(points, centers)
        labels = new_labels[:]
        new_centers = update_centers(points, labels, num_clusters)

        if has_converged(centers, new_centers):
            converged = True
        else:
            centers = new_centers[:]
            it += 1

    return (centers, labels, it)
