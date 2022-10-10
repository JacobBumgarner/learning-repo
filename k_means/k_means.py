"""K-Means Clustering algorithm code.

Author: Jacob R. Bumgarner
Email: jacobbum21@gmail.com
"""
import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(
        self,
        n_clusters: int = 3,
        centroid_init: str = "kmeans++",
        max_iterations: int = 1000,
        verbose: bool = False,
    ):
        """Construct a k-means clustering object.

        Can be fit to input data, return data labels, and return centroid positions.

        Parameters
        ----------
        n_clusters : int, optional
            The number of clusters to compute. Default is 3.
        centroid_init : str, optional
            The method used to initialized the centroids. Must be either "kmeans++" or
            "random". Default is "kmeans++".
        max_iterations : int, optional
            The maximum  number of iterations that the algorithm will run prior to
            ending the convergence. Default is 1000.
        verbose : bool, optional
            Whether to print the status of the algorithm. Defalut is False.

        Attributes
        ----------
        cluster_center : np.ndarray of shape (m_clusters, n_feature)
            The centers to the fitted clusters
        labels : np.ndarray of shape (n_samples,)
            An array containing 0-index labels for each input data point.

        """
        # First initialize random location for the clusters
        if centroid_init.lower() not in ["kmeans++", "random"]:
            raise ValueError("centroid_init should be either 'kmeans++' or 'random'")
        self.centroid_init = centroid_init

        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.verbose = verbose
        return

    def fit(self, input_data: np.ndarray):
        """Compute k-means clutsering for the input data.

        Parameters
        ----------
        input_data : np.ndarray with shape (m_samples, n_features)
            The input data to the algorithm.
        """

        # Prepare the array
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        # Find the dimensionality of the dataset
        if not input_data.ndim == 2:
            raise TypeError("The input array should only have two dimensions.")

        if self.centroid_init == "kmeans++":
            centroids = self._init_centroids_plusplus(input_data, self.n_clusters)
        else:
            centroids = self._init_centroids_random(input_data, self.n_clusters)

        # Keep track of centroid history for visualization purposes
        self.centroid_history = [centroids.copy()]

        # Now start the k_means algorithm.
        #   continue until the iterations are maxed or the centroids converge.
        iteration = 0
        old_centroids = centroids.copy() + 1
        while (
            not np.all(np.isclose(centroids, old_centroids, atol=0.001))
            and iteration < self.max_iterations
        ):
            if self.verbose:
                print(f"Iteration: {iteration}")

            # Keep track of the old centroids
            old_centroids = centroids.copy()

            # Identify the labels for each point
            data_labels = self._compute_labels(input_data, centroids)

            # Update each centroid position based on the current point labels
            centroids = self._update_centroid_locations(
                centroids, input_data, data_labels
            )

            # Update the iteration and old centroids
            iteration += 1

            self.centroid_history.append(centroids.copy())

        return

    def _init_centroids_random(
        self, input_data: np.ndarray, n_centroids: int
    ) -> np.ndarray:
        """Randomly initialize centroid points.

        Parameters
        ----------
        input_data : np.ndarray
            The input data.
        n_centroids : int
            The number of centroid points to initialize from randomly
            selected points in the input dataset.

        Returns
        -------
        centroids : np.ndarray
            The initialized centroids
        """
        rows = np.random.choice(input_data.shape[0], n_centroids, replace=False)
        centroids = input_data[rows]
        return centroids

    def _init_centroids_plusplus(
        self, input_data: np.ndarray, n_centroids: int
    ) -> np.ndarray:
        """Initialize centroid points using the kmeans++ algorithm.

        Parameters
        ----------
        input_data : np.ndarray
            The input data.
        n_centroids : int
            The number of centroid points to initialize from randomly selected
            points in the input dataset.

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
        for _ in range(1, n_centroids):
            # compute squared l2 of input_data to all centroids
            distances = cdist(input_data, np.asarray(centroids), "sqeuclidean")

            # get min distance for the centroids
            distances = distances.min(axis=1)

            # select a new centroid randomly based on the prob. distribution
            # previous centroids will have probabilities of 0 - highly unlikely
            # to get a reselection.
            prob_distribution = distances / distances.sum()
            centroid_index = np.random.choice(input_data.shape[0], p=prob_distribution)

            centroids.append(input_data[centroid_index])
            centroid_rows.append(centroid_index)

        return np.asarray(centroids)

    def _compute_labels(
        self, input_data: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
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
        distances = self._calculate_distances(input_data, centroids)

        labels = np.argmin(distances, axis=0)
        return labels

    def _calculate_distances(
        self, input_data: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """Calculate the distance of each input point to each centroid.

        Parameters
        ----------
        input_data : np.ndarray
            The input data to calculate distances between each centroid point.
        centroids : np.ndarray
            The centroid points used to calculate distances towards.

        Returns
        -------
        distances : np.ndarray with shape (m_samples, n_features).
            The distances of each input data point to each centroid point.
        """
        distances = np.zeros((centroids.shape[0], input_data.shape[0]))
        for i in range(centroids.shape[0]):
            distances[i] = np.linalg.norm(input_data - centroids[i], axis=-1)
        distances = cdist(centroids, input_data, "sqeuclidean")
        return distances

    def _update_centroid_locations(
        self, centroids: np.ndarray, input_data: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Update the location of each centroid to its center of mass.

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
            The updated centroids.
        """
        for i in range(
            centroids.shape[0]
        ):  # iterate over array to prevent div 0 errors
            if not np.any(labels == i):
                continue
            centroids[i] = np.mean(input_data[labels == i], axis=0)
        return centroids
