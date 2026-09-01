import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

np.random.seed(42)

X = np.array([
    [2, 4, 6],
    [3, 6, 9],
    [4, 8, 12],
    [5, 10, 15],
    [6, 12, 18],
    [7, 14, 21]
])

pca = PCA(n_components=2)

X_reduced = pca.fit_transform(X)

print("Original shape:")
print(X.shape)

print("Reduced shape:")
print(X_reduced.shape)

print("Principal components:")
print(pca.components_)

print("Explained variance:")
print(pca.explained_variance_)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

print("Reduced data:")
print(X_reduced)

plt.scatter(X[:, 0], X[:, 1])

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Original Data")

plt.show()

plt.scatter(
    X_reduced[:, 0],
    np.zeros(len(X_reduced))
)

plt.xlabel("Principal Component 1")
plt.title("PCA: 2D → 1D")

plt.show()

mean = X.mean(axis=0)

direction = pca.components_[0]

plt.scatter(X[:, 0], X[:, 1])

plt.quiver(
    mean[0],
    mean[1],
    direction[0],
    direction[1],
    angles="xy",
    scale_units="xy",
    scale=1
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("PCA Principal Direction")

plt.show()