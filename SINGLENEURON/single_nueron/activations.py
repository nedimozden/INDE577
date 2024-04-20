import numpy as np

from sklearn.utils import shuffle

__all__ = ['linear_regression_activation', 'sigmoid_activation']

def plot_decision_regions(X, y, clf, resolution=0.02):
    """
    Plot decision regions of a classifier.

    Parameters
    ----------
    X : numpy.ndarray
    The feature matrix.

    y : numpy.ndarray
    The target vector.

    clf : object
    Classifier object with a 'predict' method.

    resolution : float, optional
    The resolution of the grid for plotting. Default is 0.02.
    """

    # Define markers and colors for the plot
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = plt.cm.jet

    # Extract min and max values for the two features and create a meshgrid
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))

    # Predict the class labels for each combination in the grid
    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # Plot the regions
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot the data points
    for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0],
    y=X[y == cl, 1],
    alpha=0.8,
    marker=markers[idx],
    label=cl,
    edgecolor='black')


class SingleNeuron:
    """
    Represents a single artificial neuron.

    Attributes
    ----------
    activation_function : callable
    The function applied to the neuron's output (pre-activation).
    Example: sigmoid, tanh, relu, etc.

    cost_function : callable
    The loss function used to measure model performance during training.
    Example: mean squared error, cross entropy, etc.

    w_ : numpy.ndarray
    Weights vector for the neuron, with the last entry being the bias.

    errors_ : list of float
    Mean squared error for each epoch during training.

    Methods
    -------
    train(X, y, alpha=0.005, epochs=50) -> "SingleNeuron":
    Trains the neuron using stochastic gradient descent.

    predict(X: numpy.ndarray) -> numpy.ndarray:
    Predicts the output using the trained neuron.

    plot_cost_function():
    Plots the cost function across epochs.

    plot_decision_boundary(X, y, xstring="x", ystring="y"):
    Plots the decision boundary if the neuron is used for binary classification.
    """

    def __init__(self, activation_function: callable, cost_function: callable):
        self.activation_function = activation_function
        self.cost_function = cost_function
        self.w_ = None
        self.errors_ = []

    def train(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.005, epochs: int = 50) -> "SingleNeuron":
        """
        Trains the neuron on the provided data.

        Parameters
        ----------
        X : numpy.ndarray
        The feature matrix with shape (number_samples, number_features).

        y : numpy.ndarray
        The target vector with shape (number_samples,).

        alpha : float, optional
        The learning rate. Default is 0.005.

        epochs : int, optional
        Number of times the entire dataset is shown to the model. Default is 50.

        Returns
        -------
        SingleNeuron
        The trained neuron.
        """
        self.w_ = np.random.randn(1 + X.shape[1])
        N = X.shape[0]

        for _ in range(epochs):
        errors = 0
        for xi, target in zip(X, y):
        update = alpha * (self.predict(xi) - target)
        self.w_[:-1] -= update * xi
        self.w_[-1] -= update
        errors += self.cost_function(self.predict(xi), target)
        self.errors_.append(errors / N)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the output for the given input data.

        Parameters
        ----------
        X : numpy.ndarray
        The feature matrix to predict on.

        Returns
        -------
        numpy.ndarray
        Predictions for the input samples.
        """
        z = np.dot(X, self.w_[:-1]) + self.w_[-1]
        return self.activation_function(z)

    def plot_cost_function(self):
        """Plots the cost across epochs."""
        fig, axs = plt.subplots(figsize=(10, 8))
        axs.plot(range(1, len(self.errors_) + 1), self.errors_, label="Cost function")
        axs.set_xlabel("Epochs", fontsize=15)
        axs.set_ylabel("Cost", fontsize=15)
        axs.legend(fontsize=15)
        axs.set_title("Cost Across Epochs During Training", fontsize=18)
        plt.show()

    def plot_decision_boundary(self, X, y, xstring="x", ystring="y"):
        """
        Plots the decision boundary for 2D data.

        Parameters
        ----------
        X : numpy.ndarray
        The feature matrix.

        y : numpy.ndarray
        The target vector.

        xstring : str, optional
        Label for the x-axis. Default is "x".

        ystring : str, optional
        Label for the y-axis. Default is "y".
        """

        plt.figure(figsize=(10, 8))
        plot_decision_regions(X, y, clf=self)
        plt.title("Neuron Decision Boundary", fontsize=18)
        plt.xlabel(xstring, fontsize=15)
        plt.ylabel(ystring, fontsize=15)
        plt.legend(loc='upper left')
        plt.show()
    def sign_activation(z):
        return np.sign(z)

    def linear_regression_activation(z):
        return z

    def sigmoid_activation(z):
        return 1.0/(1.0 + np.exp(-z))
    def mean_sqaured_error(y_hat, y):
        return .5*(y_hat - y)**2

    def cross_entropy_loss(y_hat, y):
    return - y*np.log(y_hat) - (1 - y)*np.log(1 - y_hat)

    def train_test_split(X, y, percent = .15):
    X, y = shuffle(X, y)
    split_index = np.int(np.round(X.shape[0])*.15)
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[-split_index:], y[-split_index:]
    return X_train, y_train, X_test, y_test