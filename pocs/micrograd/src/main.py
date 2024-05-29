from micrograd import nn
from micrograd.engine import Value
from graphviz import Digraph

# Keep track of all Value instances
all_values = []

n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
all_values.extend(x)
y = n(x)
all_values.append(y)

dot = Digraph()
y.backward()
for v in reversed(all_values):
    if v.grad is not None:  # Only include Values that were involved in the computation
        dot.node(name=str(id(v)), label="{:.2f}".format(v.data), fillcolor='lightblue')
        for u in v._prev:
            dot.edge(str(id(u)), str(id(v)))
dot.render('graph', view=True)