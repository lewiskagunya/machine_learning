import numpy as np 

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

y = np.array([
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80
])
X_new = np.array([4.5])
K = 3
distances = np.abs(X.flatten() - X_new)

print(distances)
nearest_indices = np.argsort(distances)[:K]
nearest_y = y[nearest_indices]
prediction = np.mean(nearest_y)

print(prediction)