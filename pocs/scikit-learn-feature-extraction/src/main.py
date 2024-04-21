from sklearn.feature_extraction import DictVectorizer

measurements = [
     {'city': 'Dubai', 'temperature': 33.},
     {'city': 'London', 'temperature': 12.},
     {'city': 'San Francisco', 'temperature': 18.},
]

vec = DictVectorizer()
vec.fit_transform(measurements).toarray()

features_names = vec.get_feature_names_out()
print(features_names)

# plot features
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.bar(features_names, vec.fit_transform(measurements).toarray()[0])
plt.show()
