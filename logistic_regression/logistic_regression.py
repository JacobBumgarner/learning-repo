import warnings

import numpy as np

# import tensorflow as tf
import pandas as pd
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("logistic_regression/heart.csv")

## IMPORTANT ##
# The targets are flipped. Let's correct that
df.target = df.target.replace({0: 1, 1: 0})

# 1. Split the dataset into train/test datasets.
targets = df.pop("target")

x_train, x_test, y_train, y_test = train_test_split(
    df, targets, test_size=0.25, random_state=42
)
y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

# 2. Normalize the variables with continuous data with parameters from the train set
features_to_standardize = ["age", "trestbps", "chol", "thalach", "oldpeak"]

column_transformer = ColumnTransformer(
    [("scaler", StandardScaler(), features_to_standardize)], remainder="passthrough"
)
x_train = column_transformer.fit_transform(x_train)
x_test = column_transformer.transform(x_test)


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
        self.weights = np.zeros((n_input_features, 1))  # column matches X rows
        self.bias = 0

        self.fit = False  # indicates the training state of the classifier

    def linear_transform(self, X) -> np.ndarray:
        """Apply a linear transformation to the input data with the weights and bias.

        Args:
            X (np.ndarray): The data to be transformed.

        Returns:
            np.ndarray: The linearly transformed data.
        """

        Z = np.matmul(X, self.weights) + self.bias
        return Z

    def sigmoid(self, Z) -> np.ndarray:
        """Apply the sigmoid function.

        Args:
            Z (np.ndarray): The linear data to be transformed with the sigmoid function.

        Returns:
            np.ndarray: The sigmoid-transformed data.
        """
        A = 1 / (1 + np.exp(-Z))
        return A

    def compute_binary_cross_entropy_cost(self, A: np.ndarray, Y: np.ndarray) -> float:
        """Compute the binary cross entropy cost of a labeled dataset and predictions

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
        # First compute dA with the derivative of the cost function
        dA = -1 * (Y / A) + ((1 - Y) / (1 - A))

        # # Then compute dZ
        dZ = A * (1 - A)

        # Then compute dW and dB and apply the chain rule
        dW = np.mean(X * dZ * dA, axis=0, keepdims=True).T
        dB = np.mean(1 * dZ * dA, axis=0, keepdims=True).T

        self.weights -= dW * learning_rate
        self.bias -= dB * learning_rate
        return

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        refit: bool = False,
        verbose: bool = False,
    ) -> np.ndarray:
        """Fit the logistic regression model with training data and target labels.

        Args:
            X (np.ndarray): The training datat..
            Y (np.ndarray): The targets of the training data.
            epochs (int, optional): The number of training iterations. Defaults to 100.
            refit (bool, optional). Whether or not to refit the model.
                Defaults to False.
            verbose (bool, optional): Print the status of the training.

        Returns:
            np.ndarray: The cost history of the training.
        """
        # Check whether the model has been fit.
        if self.fit and not refit:
            warnings.warn("Warning: This model has already been fit.")

        if not X.shape[-1] == self.weights.shape[0]:
            raise ValueError(
                "The shape of the last axis of the training data must match the shape "
                "of the data used to instantiate the model."
            )

        if Y.ndim == 1:
            Y = np.expand_dims(Y, axis=1)

        # Fit the model
        costs = []
        for i in range(epochs):
            # First conduct linear transformation
            Z = self.linear_transform(X)

            # Then conduct the sigmoid transformation
            A = self.sigmoid(Z)

            # Compute the cost
            cost = self.compute_binary_cross_entropy_cost(A, Y)

            # Then run the backprop
            self.backpropagate(X, A, Y, learning_rate=learning_rate)

            if verbose:
                print(f"Epoch: {i}, Cost: {cost: 0.2f}", end="\r")

            costs.append(cost)

        if verbose:
            print(f"Final model cost: {cost:0.2f}")

        self.fit = True
        return np.array(cost)

    def predict(self, X) -> np.ndarray:
        Z = self.linear_transform(X)
        A = self.sigmoid(Z)
        return (A >= 0.5).astype(int).T[0]


model = LogisticRegression(n_input_features=x_train.shape[-1])
costs = model.train(x_train, y_train, learning_rate=0.1, epochs=20, verbose=True)

print(x_test.shape)

preds = model.predict(x_test)
result = preds == y_test
accuracy = f"{result.sum() / result.shape[0] * 100: 0.2f}"
print(f"Model prediction accuracy: {accuracy}%")
