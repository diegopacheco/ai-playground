from pycaret.datasets import get_data
from pycaret.classification import ClassificationExperiment

data = get_data('diabetes')
s = ClassificationExperiment()
s.setup(data, target = 'Class variable', session_id = 123)
best = s.compare_models()

print(best)