import pytensor
from pytensor import tensor as pt

# Declare two symbolic floating-point scalars
a = pt.dscalar("a")
b = pt.dscalar("b")

# Create a simple example expression
c = a + b

# Convert the expression into a callable object that takes `(a, b)`
# values as input and computes the value of `c`.
f_c = pytensor.function([a, b], c)

assert f_c(1.5, 2.5) == 4.0
print(f_c(1.5, 2.5))