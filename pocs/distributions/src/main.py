import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, bernoulli, binom, poisson, expon, uniform, laplace, gamma, beta, dirichlet, multinomial, t, chi2, rayleigh, weibull_min, weibull_max
from scipy.stats import dirichlet

# Real-world data (example)
data = np.random.normal(loc=0, scale=1, size=1000)  # Gaussian

# Plot distributions
plt.figure(figsize=(10, 6))
plt.hist(data, density=True, alpha=0.5, label='Gaussian')

x = np.linspace(-5, 5, 100)
plt.plot(x, norm.pdf(x, loc=0, scale=1), label='Gaussian PDF')

x = np.arange(0, 1, 0.1)
plt.plot(x, bernoulli.pmf(x, p=0.5), label='Bernoulli PMF')

x = np.arange(0, 10, 1)
plt.plot(x, binom.pmf(x, n=10, p=0.5), label='Binomial PMF')

x = np.arange(0, 10, 1)
plt.plot(x, poisson.pmf(x, mu=3), label='Poisson PMF')

x = np.linspace(0, 5, 100)
plt.plot(x, expon.pdf(x, scale=1), label='Exponential PDF')

x = np.linspace(0, 1, 100)
plt.plot(x, uniform.pdf(x, loc=0, scale=1), label='Uniform PDF')

x = np.linspace(-5, 5, 100)
plt.plot(x, laplace.pdf(x, loc=0, scale=1), label='Laplace PDF')

x = np.linspace(0, 5, 100)
plt.plot(x, gamma.pdf(x, a=2, scale=1), label='Gamma PDF')

x = np.linspace(0, 1, 100)
plt.plot(x, beta.pdf(x, a=2, b=2), label='Beta PDF')

x = np.array([[5, 5], [6, 4], [4, 6], [7, 3], [3, 7], [8, 2], [2, 8], [9, 1], [1, 9], [10, 0], [0, 10]])
plt.plot(x[:, 0], multinomial.pmf(x, n=10, p=[0.5, 0.5]), label='Multinomial PMF')

x = np.linspace(-5, 5, 100)
plt.plot(x, t.pdf(x, df=2), label="Student's t PDF")

x = np.linspace(0, 10, 100)
plt.plot(x, chi2.pdf(x, df=2), label='Chi-squared PDF')

x = np.linspace(0, 5, 100)
plt.plot(x, rayleigh.pdf(x, scale=1), label='Rayleigh PDF')

# Dirichlet
alpha = [1, 1, 1]
x = np.random.dirichlet(alpha, size=1)
print(f"Random variates from Dirichlet distribution: {x}")

# Weibull_min
x = np.linspace(0, 2, 100)
plt.plot(x, weibull_min.pdf(x, c=1.5), label='Weibull Min PDF')

# Weibull_max
plt.plot(x, weibull_max.pdf(x, c=1.5), label='Weibull Max PDF')

plt.legend()
plt.show()