import numpy as np

X = np.array([
    [1, 2],
    [1, 3],
    [2, 2],
    [8, 8],
    [9, 8],
    [8, 9]
])
K = 2

centroids = np.array([
    [1, 2],
    [8, 8]
])

for iteration in range(10):

    distances = np.linalg.norm(
        X[:, np.newaxis] - centroids,
        axis=2
    )

    labels = np.argmin(distances, axis=1)

    for k in range(K):

        points = X[labels == k]

        centroids[k] = np.mean(
            points,
            axis=0
        )


print("Labels:")
print(labels)

print("\nCentroids:")
print(centroids)