import numpy as np

x = np.arange(15, dtype=np.int64).reshape(3, 5)
print(x)
x[1:, ::2] = -99
print(x)
print(x.max(axis=1))

rng = np.random.default_rng()
samples = rng.normal(size=2500)
print(samples)

