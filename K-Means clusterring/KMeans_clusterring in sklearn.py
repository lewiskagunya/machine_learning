import numpy as np 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt

X = np.array([
    [1, 2],
    [1, 3],
    [2, 2],
    [8, 8],
    [9, 8],
    [8, 9]
])

model = KMeans(n_clusters=2, n_init=10, random_state=42)
model.fit(X)

print("labels:")
labels = model.labels_
print(labels)

print("centroids:")
print(model.cluster_centers_)

new_point = np.array([[2, 3]])

prediction = model.predict(new_point)
print("prediction:")
print(prediction)


plt.scatter(X[:, 0], X[:, 1], c=labels, s=200, cmap='viridis')


plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.scatter(new_point[:, 0], new_point[:, 1], c='orange', marker='*', s=300, label='New Point')

plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title("KMeans Clustering")
plt.legend()

plt.show()