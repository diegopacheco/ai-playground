import numpy as np

a = np.arange(6)
print(f"original array {a}")

b = a.reshape(3, 2)
print(f"reshaped array \n{b}")