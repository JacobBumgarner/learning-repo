"""Logistic Regression classifier for my Medium Article.

Author: Jacob Bumgarner
Email: jacobbum21@gmail.com
"""

import numpy as np


class LogisticRegression:
    """A Logistic Regression classifier.

    Args:
        n_input_feature (int): The number of features in the dataset.

    Attributes:
        weights (np.ndarary): The weights of the model
        bias (float): The bias of the model.
        fit (bool): Whether the model has been fit to training data or not. Defaults to
            False.
    """

    def __init__(self, n_input_features: int):
        """Initialize the model."""
        self.weights = np.random.randn((n_input_features, 1))*0.01  # column matches X rows
        self.bias = np.zeros((1,1))

        self.fit = False  # indicates the training state of the classifier

    def linear_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply a linear transformation to the input data with the weights and bias.

        Args:
            X (np.ndarray): The data to be transformed.

        Returns:
            np.ndarray: The linearly transformed data.
        """

        Z = np.matmul(X, self.weights) + self.bias
        return Z

    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        """Apply the sigmoid function.

        Args:
            Z (np.ndarray): The linear data to be transformed with the sigmoid function.

        Returns:
            np.ndarray: The sigmoid-transformed data.
        """
        A = 1 / (1 + np.exp(-Z))
        return A

    def compute_cross_entropy_cost(self, A: np.ndarray, Y: np.ndarray) -> float:
        """Compute the model's cross-entropy cost for a labeled dataset.

        Args:
            A (np.ndarray): The projected probabilities for each data point.
            Y (np.ndarray): the target probabilities for each data point.

        Returns:
            float: The cost of the training iteration.
        """
        epsilon = 1e-6  # add the epsilon to prevent divide by zero warnings
        m = Y.shape[0]
        y_0 = (1 - Y) * np.log(1 - A + epsilon)  # where y target == 0
        y_1 = Y * np.log(A + epsilon)  # where y target == 1

        cost_sum = np.sum(-y_0 - y_1)
        cost = cost_sum / m
        return cost

    def backpropagate(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        learning_rate=0.01,
    ) -> None:
        """Compute a single backpropagation of the model.

        Args:
            X (np.ndarray): The array containing the training data.
            A (np.ndarray): The the predicted labels.
            Y (np.ndarray): The target labels.
            learning_rate (float, optional): The learning rate of the backprop.
                Defaults to 0.01.

        Returns:
            float: The cost of the iteration.
        """
        # Compute simplified dZ from chain rule product of dA * dZ
        dZ = (A - Y)
        
        # Then compute dW and dB and find the average loss, i.e., cost
        dW = np.mean(dZ * X, axis=0, keepdims=True).T
        dB = np.mean(dZ, axis=0, keepdims=True).T

        self.weights -= dW * learning_rate
        self.bias -= dB * learning_rate
        return

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        minibatch_size: int = 10,
        verbose: bool = False,
    ) -> np.ndarray:
        """Fit the logistic regression model with training data and target labels.

        Uses minibatch gradient descent.

        Args:
            X (np.ndarray): The training dataset.
            Y (np.ndarray): The training dataset targets.
            epochs (int, optional): The number of training iterations. Defaults to 100.
            learning_rate (float, optional): The step size of the descent.
                Defaults to 0.01.
            minibatch_size (int, optional): The size of the minibatch for the stochastic
                gradient descent. If None, runs batch gradient descent. Defaults to 10.
            verbose (bool, optional): _description_. Defaults to False.

        Raises:
            Attribute: Raises error if the model has already been fit.
            ValueError: Raises error if the number of features doesn't match the
                instantiated feature count.

        Returns:
            np.ndarray: The cost history.
        """
        # Check whether the model has been fit.
        if self.fit:
            raise AttributeError("Error: This model has already been fit.")
        self.fit = True

        if not X.shape[-1] == self.weights.shape[0]:
            raise ValueError(
                "The shape of the last axis of the training data must match the shape "
                "of the data used to instantiate the model."
            )

        if Y.ndim == 1:
            Y = np.expand_dims(Y, axis=1)

        # Fit the model
        costs = []
        accuracies = []
        weight_hist, bias_hist = [], []
        for i in range(epochs):
            weight_hist.append(self.weights[:, 0].copy())
            bias_hist.append(self.bias.copy())
            
            # Isolate the minibatch
            if minibatch_size:
                batch_indices = np.random.choice(
                    X.shape[0], size=minibatch_size, replace=False
                )
                X_batch, Y_batch = X[batch_indices], Y[batch_indices]
            else:
                X_batch, Y_batch = X, Y

            # Compute the linear transformation
            Z = self.linear_transform(X_batch)

            # Compute the sigmoid transformation
            A = self.sigmoid(Z)

            # Compute the cost
            cost = self.compute_cross_entropy_cost(A, Y_batch)

            # Then run the backprop
            self.backpropagate(X_batch, A, Y_batch, learning_rate=learning_rate)

            if verbose:
                print(f"Epoch: {i}, Cost: {cost: 0.2f}          ", end="\r")

            costs.append(cost)
            accuracies.append(self.accuracy(self.predict(X), Y[:, 0]))

        if verbose:
            print(f"Final model cost: {cost:0.2f}              ")

        self.fit = True
        return np.array(costs), np.array(accuracies), np.array(weight_hist).T, np.array(bias_hist).T[0, 0]

    def predict(self, X) -> np.ndarray:
        """Predict the labels for a set of input data.

        Args:
            X (np.ndarray): The data for label predictions.

        Returns:
            np.ndarray: The predictions for each sample.
        """
        if not self.fit:
            raise AttributeError(
                "Error: This classifier has not been fit to any training data."
            )
        Z = self.linear_transform(X)
        A = self.sigmoid(Z)
        return A.T[0]

    def accuracy(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Return the accuracy of a dataset prediction.

        Args:
            predictions (np.ndarray): The output predictions from the `predict`
                function.
            labels (np.ndarray): The true labels to compare to the predictions.

        Returns:
            float: The prediction accuracy.
        """
        overlap = (predictions >= 0.5) == labels
        accuracy = overlap.sum() / predictions.shape[0] * 100
        return accuracy
