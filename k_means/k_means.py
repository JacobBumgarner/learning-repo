"""K-Means Clustering algorithm code.

Author: Jacob R. Bumgarner
Email: jacobbum21@gmail.com
"""
import numpy as np
from scipy.spatial.distance import cdist


def update_centroid_locations(
    centroids: np.ndarray, input_data: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Update the location of each centroid to the center of mass of its labeled points.

    Parameters
    ----------
    centroids : np.ndarray
        The centroids from the previous iteration.
    input_data : np.ndarray
        The input data to the algorithm.
    labels : np.ndarray
        The labels showing the closet centroid to each piece of input data.

    Returns
    -------
    centroids : np.ndarray
        The updated centroids based on the center of mass of the current labeled points.
    """
    for i in range(centroids.shape[0]):
        if not np.any(labels == i):
            continue

        # old_centroid = centroids[i].copy()
        # new_centroid = np.mean(input_data[labels == i], axis=0)
        # vector = new_centroid - old_centroid

        # if np.all(np.isclose(old_centroid, new_centroid, atol=0.1)):
        #     centroids[i] = new_centroid
        # else:
        #     updated = old_centroid + (vector / np.linalg.norm(vector) * 0.1)
        #     centroids[i] = updated
        centroids[i] = np.mean(input_data[labels == i], axis=0)
    return centroids


def get_labels(input_data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return the index of the closet cluster for each input data point.

    Parameters
    ----------
    input_data : np.ndarray
        The input data to find labels for.
    centroids : np.ndarray
        The centroid points used to label each input data point.

    Returns
    -------
    data_labels : np.ndarray
        The index of the closest centroid to each input point
    """
    # Compute the distance of each point to each centroid,
    #   assign each point the label of the closest centroid
    distances = calculate_distances(input_data, centroids)

    labels = np.argmin(distances, axis=0)
    return labels


def calculate_distances(input_data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Calculate the distance of each input point to each centroid.

    Parameters
    ----------
    input_data : np.ndarray
        The input data to calculate distances between each centroid point.
    centroids : np.ndarray
        The centroid points used to calculate distances towards.

    Returns
    -------
    distances : np.ndarray
        The distances of each input data point to each centroid point. Has shape
        ``(m, n)``, where ``m`` is the number of clusters, and ``n`` is the number of
        input data points.
    """
    # distances = np.zeros((centroids.shape[0], input_data.shape[0]))
    # for i in range(centroids.shape[0]):
    #     distances[i] = np.linalg.norm(input_data - centroids[i], axis=-1)
    distances = cdist(input_data, centroids, "sqeuclidean")
    return distances


def init_centroids_random(input_data: np.ndarray, n_centroids: int) -> np.ndarray:
    """Initialize centroid points.

    Parameters
    ----------
    input_data : np.ndarray
        The input data.
    n_centroids : int
        The number of centroid points to initialize from randomly selected points in
        the input dataset.

    Returns
    -------
    centroids : np.ndarray
        The initialized centroids
    """
    rows = np.random.choice(input_data.shape[0], n_centroids, replace=False)
    centroids = input_data[rows]
    return centroids


def init_centroids_plusplus(input_data: np.ndarray, n_centroids: int) -> np.ndarray:
    """Initialize centroid points using the kmeans++ algorithm.

    Parameters
    ----------
    input_data : np.ndarray
        The input data.
    n_centroids : int
        The number of centroid points to initialize from randomly selected points in
        the input dataset.

    Returns
    -------
    centroids : np.ndarray
        The initialized centroids
    """
    centroids = []
    centroid_rows = []
    
    # randomly select first centroid
    centroid_rows.append(np.random.choice(input_data.shape[0]))
    centroids.append(input_data[centroid_rows[0]])
    
    # Select other centroids
    for i in range(1, n_centroids):
        # compute squared l2 of input_data to all centroids
        distances = cdist(input_data, np.asarray(centroids), 'sqeuclidean')
        
        # get min distance for the centroids
        distances = distances.min(axis=1)

        # create probability distribution
        prob_distribution = distances / distances.sum()
        while True:
            # select a new centroid randomly based on the generated prob dist.
            row = np.random.choice(input_data.shape[0], p=prob_distribution)
            if row not in centroid_rows:  # don't reuse centroids
                centroids.append(input_data[row])
                centroid_rows.append(row)
            break

    return np.asarray(centroids)


def k_means(
    input_data: np.ndarray,
    n_clusters: int = 3,
    centroid_init: str = "kmeans++",
    max_iteration: int = 1000,
    verbose: bool = False,
) -> np.ndarray:
    """Conduct the k-means clustering algorithm and return the specified n centroids.

    Based on the input cluster count, the function will return centroid location of the
    n clusters.

    Parameters
    ----------
    input_data : np.ndarray
        The input data to the algorithm. Should be in ``(m, n)`` shape, where ``m`` is
        the number of samples, and ``n`` is the dimensionality of the input dataset.
    n_clusters : int, optional
        The number of clusters to compute, defaults to 3.
    centroid_init : str, optional
        The method used to initialized the centroids. Must be either "kmeans++" or
        "random". Default is "kmeans++".
    max_iterations : int, optional
        The maximum  number of iterations that the algorithm will run prior to ending
        the convergence, defaults to 1000.
    verbose : bool, optional
        Whether to print the status of the algorithm. Defalut is False.

    Returns
    -------
    centroids : np.ndarray
        The centroid positions of the clusters with (m, n) dimensionality.
    """
    # Prepare the array
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)

    # Find the dimensionality of the dataset
    if not input_data.ndim == 2:
        raise TypeError("The input array should only have two dimensions.")

    # First initialize random location for the clusters
    if centroid_init == "kmeans++":
        centroids = init_centroids_plusplus(input_data, n_clusters)
    else:
        centroids = init_centroids_random(input_data, n_clusters)

    # Keep track of centroid history for visualization purposes
    centroid_history = [centroids.copy()]

    # Now start the k_means algorithm.
    #   continue until the iterations are maxed or the centroids converge.
    iteration = 0
    old_centroids = centroids.copy() + 1
    # while np.any(centroids != old_centroids) and iteration < max_iteration:
    while (
        not np.all(np.isclose(centroids, old_centroids, atol=0.001))
        and iteration < max_iteration
    ):
        if verbose:
            print(f"Iteration: {iteration}")

        # Keep track of the old centroids
        old_centroids = centroids.copy()

        # Identify the labels for each point
        data_labels = get_labels(input_data, centroids)

        # Update the location of each centroid based on the current point labels
        centroids = update_centroid_locations(centroids, input_data, data_labels)

        # Update the iteration and old centroids
        iteration += 1

        centroid_history.append(centroids.copy())

    return centroid_history
