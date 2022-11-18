from enum import Flag
from operator import index
import numpy as np  # imports a fast numerical programming library
import scipy as sp  # imports stats functions, amongst other things
import matplotlib as mpl  # this actually imports matplotlib
import matplotlib.cm as cm  # allows us easy access to colormaps
import matplotlib.pyplot as plt  # sets up plotting under plt
import pandas as pd  # lets us handle data as dataframes
import jax.numpy as jnp  # ai-accelerator agnostic numpy implementations
import jax.scipy as jsp  # ai-accelerator agnostic scipy implementations
import jax  # autograd + XLA
from tensorflow_probability.substrates import jax as tfp  # tensorflow probability
import seaborn as sns
import warnings
from jax import random, jit, value_and_grad
from jax.config import config
from scipy.optimize import minimize
warnings.filterwarnings("ignore")


class kernel_functions:
    """
    The kernel function lifts the input space to the feature space.
    """

    def rbf_kernel(x_a: float, x_b: float, theta=[1.0, 1.0]):
        """
        The covariance function k(x_a, x_b) models the joint variability of
        the Gaussian Process random variables. Here, we take it as the 
        exponentiated quadratic covariance function, called the RBF kernel.

        theta[1]: Controls vertical variation of the process.
        theta[0]: Controls smoothness of the functions.
        """

        sqdist = jnp.sum(x_a ** 2, 1).reshape(-1, 1) + \
            jnp.sum(x_b ** 2, 1) - 2 * jnp.dot(x_a, x_b.T)
        return theta[1] ** 2 * jnp.exp(-0.5 / theta[0] ** 2 * sqdist)
        # Squared euclidean distance, taken pairwise across all elements of x_a and x_b.

    def periodic_kernel(x_a: float, x_b: float, sigma=1.0):
        return 0

    def linear_kernel(x_a: float, x_b: float, sigma=1.0):
        return 0

    def rbf_kernel_diag(dim, theta=[1.0, 1.0]):
        return jnp.full(shape=dim, fill_value=theta[1] ** 2)


class preprocess:
    """
    Functions for pre-processing data before feeding to kernel, etc.
    """

    def __init__(self):
        pass

    def jitter_matrix(self, dim_count: int, value=1e-5):
        """
        Add jitter along diagonals to make sure matrix doesn't become singular.
        Reduces overfitting in many huge-input cases.
        """

        return jnp.eye(dim_count) * value

    def softplus(self, X):
        """
        Activation function, smoothened ReLU, f(x) = log (1 + exp(x)).
        """
        return jnp.log(1 + jnp.exp(X))

    def softplus_inv(self, X):
        """
        Inverted softplus, f(x) = log(exp(x) - 1).
        """
        return jnp.log(jnp.exp(X) - 1)

    def pack_params(self, theta, X_m):
        return jnp.concatenate([self.softplus_inv(theta), X_m.ravel()])

    def unpack_params(self, params):
        return self.softplus(params[:2]), jnp.array(params[2:].reshape(-1, 1))

    def softplus(self, X, inv=False):
        """
        Constrains output to always be positive.
        """
        if inv:
            return jnp.log(1 + jnp.exp(X))
        else:
            return jnp.log(-1 + jnp.exp(X))

    def pack(theta, X_m):
        pre = preprocess()
        return jnp.concatenate([pre.softplus_inv(theta), X_m.ravel()])

    def unpack(params):
        pre = preprocess()
        return pre.softplus(params[:2]), jnp.array(params[2:].reshape(-1, 1))

    def woodbury_inversion(self, A, U, C, V):
        """
        Woodbury inversion implementation.

        Parameters
        ----------
        Ainv: nparray
        Inverse of the matrix A.

        U: nparray
        The matrix U.

        Cinv: nparray
        Inverse of the matrix C.

        V: nparray
        The matrix V.
        """

        return np.linalg.inv(A) - np.dot(np.dot(np.linalg.inv(A), U), np.dot(np.linalg.inv(np.linalg.inv(C + np.dot(V, np.dot(np.linalg.inv(A), U)))), np.dot(V, np.linalg.inv(A))))


class variational:
    """
    Functions for Variational Inference, including lower-bounds, etc.
    """

    def nlb_fn(X, y, sigma_y):
        n = X.shape[0]

        def nlb(parameters):
            """
              Negative lower bound on log marginal likelihood.

            Parameters:
                theta: Kernel Parameters
                X_m: Inducing points

            Variables:
                sigma_y: Noise parameters
                X, y: All data available.

            """
            n = X.shape[0]
            pre = preprocess()

            theta, X_m = preprocess.unpack(parameters)

            K_mm = kernel_functions.rbf_kernel(
                X_m, X_m, theta) + pre.jitter_matrix(X_m.shape[0])
            K_mn = kernel_functions.rbf_kernel(X_m, X, theta)

            L = jnp.linalg.cholesky(K_mm)  # m x m
            A = jsp.linalg.solve_triangular(
                L, K_mn, lower=True) / sigma_y  # m x n
            AAT = A @ A.T  # m x m
            B = jnp.eye(X_m.shape[0]) + AAT  # m x m
            LB = jnp.linalg.cholesky(B)  # m x m
            c = jsp.linalg.solve_triangular(
                LB, A.dot(y), lower=True) / sigma_y  # m x 1

            lb = - n / 2 * jnp.log(2 * jnp.pi)
            lb -= jnp.sum(jnp.log(jnp.diag(LB)))
            lb -= n / 2 * jnp.log(sigma_y ** 2)
            lb -= 0.5 / sigma_y ** 2 * y.T.dot(y)
            lb += 0.5 * c.T.dot(c)
            lb -= 0.5 / sigma_y ** 2 * \
                jnp.sum(jnp.full(shape=n, fill_value=theta[1] ** 2))
            lb += 0.5 * jnp.trace(AAT)

            return -lb[0, 0]

        nlb_gradient = jit(value_and_grad(nlb))

        def nlb_grad_wrapper(parameters):
            value, grads = nlb_gradient(parameters)
            return np.array(value), np.array(grads)

        return nlb_grad_wrapper

    @jit
    def optimizer(theta, X_m, X, y, sigma_y):
        """Optimize mu_m and A_m"""

        pre = preprocess()

        precision = (1.0 / sigma_y ** 2)

        K_mm = kernel_functions.rbf_kernel(
            X_m, X_m, theta) + pre.jitter_matrix(X_m.shape[0])
        K_mm_inv = jnp.linalg.inv(K_mm)
        K_nm = kernel_functions.rbf_kernel(X, X_m, theta)
        K_mn = K_nm.T

        Sigma = jnp.linalg.inv(K_mm + precision * K_mn @ K_nm)

        mu_m = precision * (K_mm @ Sigma @ K_mn).dot(y)
        A_m = K_mm @ Sigma @ K_mm

        return mu_m, A_m, K_mm_inv
