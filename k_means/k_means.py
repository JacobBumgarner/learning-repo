"""K-Means Clustering algorithm code.

Author: Jacob R. Bumgarner
Email: jacobbum21@gmail.com
"""
import numpy as np


def update_centroid_locations(
    centroids: np.ndarray, 
    input_data: np.ndarray, 
    labels: np.ndarray
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
        if not np.any(labels==i):
            continue
        centroids[i] = np.mean(
            input_data[labels==i], axis=0
        )
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
    distances = np.zeros((centroids.shape[0], input_data.shape[0]))
    for i in range(centroids.shape[0]):
        distances[i] = np.linalg.norm(input_data-centroids[i], axis=-1)
    return distances


def k_means(
    input_data: np.ndarray, n_clusters: int = 2, max_iteration: int = 1000
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
        The number of clusters to compute, by default 3.

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
    centroids = np.random.uniform(
        input_data.min(), input_data.max(), (n_clusters, input_data.shape[-1])
    )

    # Now start the k_means algorithm.
    #   continue until the iterations are maxed or the centroids converge.
    iteration = 0
    old_centroids = centroids.copy() + 1  # 
    while np.any(centroids != old_centroids) and iteration < max_iteration:
        # Keep track of the old centroids
        old_centroids = centroids.copy()

        # Identify the labels for each point
        data_labels = get_labels(input_data, centroids)
        
        # Update the location of each centroid based on the current point labels
        centroids = update_centroid_locations(centroids, input_data, data_labels)

        # Update the iteration and old centroids
        iteration += 1

    return centroids
