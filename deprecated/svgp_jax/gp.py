from operator import index
import numpy as np  # imports a fast numerical programming library
import scipy as sp  # imports stats functions, amongst other things
import matplotlib as mpl  # this actually imports matplotlib
import matplotlib.cm as cm  # allows us easy access to colormaps
import matplotlib.pyplot as plt  # sets up plotting under plt
import pandas as pd  # lets us handle data as dataframes
import jax.numpy as jnp  # ai-accelerator agnostic numpy implementations
import jax.scipy as jsp
import jax  # autograd + XLA
from tensorflow_probability.substrates import jax as tfp  # tensorflow probability
import seaborn as sns
import warnings
from jax import random, jit, value_and_grad
from kernel import kernel_functions, preprocess
warnings.filterwarnings("ignore")


class gaussian_process:
    """
    Non-linear regression by infinite-parameter Gaussian distributions over functions.
    Make predictions based on Gaussian Process prior and N observed data points: (X1, y1).

    Parameters
    --------------
    X1, Y1 : float | nparray
        Arrays with floating point numbers representing input data.

    """

    def __init__(self, kernel_function_name="rbf"):
        pre = preprocess()
        if kernel_function_name != "rbf":
            pass
        else:
            self.kernel_function = kernel_functions.rbf_kernel

        self.max = 3
        self.min = -3

    def gaussian_prior(self, x_test, theta = [1,1], sample_count=100):
        pre = preprocess()

        K = self.kernel_function(x_test, x_test, theta)
        # The Kernel function applied over itself resembles the variance of the distribution.
        L = jnp.linalg.cholesky(K + pre.jitter_matrix(len(x_test)))
        # The Cholesky Decomposition essentially gives the square-root of the matrix, so from variance to SD.
        gp_prior = jnp.dot(L, np.random.normal(
            size=(len(x_test), sample_count)))

        # N; N' = mu' + sig' * N

        def rescale(y: float) -> float:
            return self.min + (self.max - self.min)/len(gp_prior)*y

        gp_prior_dict = []

        for _ in range(len(gp_prior)):
            gp_prior_dict.append({"X": rescale(_), "Y": gp_prior[_]})

        gp_prior = {}

        x = []
        all_y = []
        for _ in gp_prior_dict:
            x.append(_["X"])
            all_y.append(_["Y"])

        gp_prior = {"X": x, "Y": all_y}

        return gp_prior

    def gaussian_posterior(self, gp_prior, X_data, Y_data, theta = [1,1], X_test=[]):
        """
        Find the posterior Gaussian Process with a Gaussian Process prior.
        Generate posterior = p(y2 | X2, X1, y1), where X1 and y1 are
        observed data points, X2 is the input to predict on.

        Output
        ----------

        Returns an array of numbers based on the shape of the Gaussian Process
        prior, with the following specifications.

        gp_posterior: array
            A horizontal matrix, of size number_of_samples x points_inferenced

        f_prior: array
            A vertical matrix, of size points_inference x number_of_samples

        Parameters
        ----------
        kernel_function: function
            The kernel function (only RBF implemented till now), with some
            preset sigma to generate the Gaussian Process prior.

        X_data: float | nparray
            Numbers on the X-axis real-line where we know the value of f(X).

        Y_data: float | nparray
            Numbers on the Y-axis real-line where we know the value of f(X)
            corresponding to X.

        """
        pre = preprocess()
        N = len(X_data)
        n, f_num = gp_prior.shape

        K = self.kernel_function(X_data, X_data, theta = theta)
        L = np.linalg.cholesky(
            K + pre.jitter_matrix(dim_count=N, value=0.0001))
        K_ = self.kernel_function(X_test, X_test)
        Lk = np.linalg.solve(L, self.kernel_function(X_data, X_test))
        mu = np.dot(Lk.T, np.linalg.solve(L, Y_data))
        L = np.linalg.cholesky(
            K_ + pre.jitter_matrix(dim_count=n) - np.dot(Lk.T, Lk))
        f_post = mu.reshape(-1, 1) + np.dot(L, np.random.normal(size=(n, 300)))

        def rescale(y: float) -> float:
            return self.min + (self.max - self.min)/len(gp_prior)*y

        gp_posterior_dict = []

        for _ in range(len(f_post)):
            gp_posterior_dict.append({"X": rescale(_), "Y": f_post[_]})

        gp_post = {}

        x = []
        all_y = []
        for _ in gp_posterior_dict:
            x.append(_["X"])
            all_y.append(_["Y"])

        gp_post = {"X": x, "Y": all_y}

        gp_posterior = [[]]

        std = np.sqrt(np.diag(K_) - np.sum(Lk**2, axis=0))

        for func_index in range(f_num):
            gp_posterior.append([])
            for _ in range(n):
                gp_posterior[func_index].append(gp_post["Y"][_][func_index])

        return gp_posterior[:-1], gp_post, mu, std
