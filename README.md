[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)



<div align = center>
<a href = "github.com/plugyawn"><img width="300px" height="300px" src= "https://user-images.githubusercontent.com/76529011/202820671-44b1d06e-1d92-4585-b2d8-f3e18fad50cb.png"></a>
</div>

-----------------------------------------
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)![Compatibility](https://img.shields.io/badge/compatible%20with-python3.6.x-blue.svg)

```GP-Zoo``` is a collection of implementations of significant papers on Gaussian Processes with readable, simple code, over the past decade; ranging from Sparse Gaussian Process Regression from Titsias, 2009, to Stochastic Variational Gaussian Processes for classification and regression, by Hensman, 2014. 

# Implementations

We start with the standard ```GPJax``` dataset for regression.
```python

import jax.random as jr
key = jax.random.PRNGKey(0)
n = 1000
noise = 0.2

x = jr.uniform(key=key, minval=-5.0, maxval=5.0,
               shape=(n,)).sort().reshape(-1, 1)


def f(x): return jnp.sin(4 * x) + jnp.cos(2 * x)


signal = f(x)
y = signal + jr.normal(key, shape=signal.shape) * noise

xtest = jnp.linspace(-5.5, 5.5, 500).reshape(-1, 1)

z = jnp.linspace(-5.0, 5.0, 50).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, y, "o", alpha=0.3, label="Samples")
ax.plot(xtest, f(xtest), label="True function")
[ax.axvline(x=z_i, color="black", alpha=0.3, linewidth=1) for z_i in z]
plt.show()
```
![image](https://user-images.githubusercontent.com/76529011/202865518-e7ee8de9-7b8a-4f70-b883-55fc3b68a3a7.png)

```GP-Zoo```'s regression implementations are all based on this dataset, although it can be easily switched out by replacing ```x``` and ```y``` with a different dataset.
For example, Stochastic Variational Gaussian Processes for Classification (*Hensman et al, 2014*), classifies the ```moons``` dataset:
```python
n_samples = 100
noise = 0.1
random_state = 0
shuffle = True

X, y = make_moons(
    n_samples=n_samples, random_state=random_state, noise=noise, shuffle=shuffle
)
X = StandardScaler().fit_transform(X)  # Yes, this is useful for GPs

X, y = map(jnp.array, (X, y))

plt.scatter(X[:, 0], X[:, 1], c=y)
```
![image](https://user-images.githubusercontent.com/76529011/202865498-775da1e2-de64-4bd9-af97-f578cc0aa639.png)

# Examples of regression implementations

## Stochastic Variational Gaussian Process
A parallelizable, scalable algorithm for Gaussian Processes, optimized for dealing with big data. Crosses are inducing points, while translucent blue dots are the original full dataset.

![image](https://user-images.githubusercontent.com/76529011/202865852-698638d7-c08b-4afb-99cf-cb7e3ec9bf94.png)

## Sparse Gaussian Process

Based on the same dataset, a sparse Gaussian Process would include:
![image](https://user-images.githubusercontent.com/76529011/202865600-7f74d040-dcfd-4be2-8ad4-06d69454cc8d.png)

## Titsias's Sparse Gaussian Process

Based on Titsias et al (2009), the greedy-algorithm for selecting inducing points leads to a good approximation of an exact Gaussian Process.
![image](https://user-images.githubusercontent.com/76529011/202865698-503ff319-5c21-48ca-8aad-68b90943d40c.png)

## Variational training of Inducing points

A technique based on minimzing the Evidence Lower Bound as a loss for how well the inducing points approximate the full dataset.
![image](https://user-images.githubusercontent.com/76529011/202865728-c0570d98-f666-4f49-8883-69c615ad8587.png)



# Contributing

GP-Zoo is a work-in-progress, so feel free to put in a PR and share in the work!

Notably, if you can find a way to add backtesting strategies through boolean expressions passed to the function, that would be really helpful!
Reach me at ```progyan.das@iitgn.ac.in``` or ```progyan.me```.



