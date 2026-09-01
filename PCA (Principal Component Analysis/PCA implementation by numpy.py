import numpy as np 

X = np.array([
    [2, 4, 6],
    [3, 6, 9],
    [4, 8, 12],
    [5, 10, 15],
    [6, 12, 18],
    [7, 14, 21]
])

mean = np.mean(X, axis=0)

print("mean:")
print(mean)

x_centered = X - mean 
print("center points:")
print(x_centered)

covariance = (x_centered.T @ x_centered)/(len(x_centered))
print("covariance:")
print(covariance)

eigenvalues,eigenvectors = np.linalg.eigh(covariance)
print("eigenvalues:")
print(eigenvalues)

print("eigenvectors:")
print(eigenvectors)

idx = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[idx]

eigenvectors = eigenvectors[:, idx]

n_components = 1
components = eigenvectors[:, :n_components]

x_reduced =  x_centered @ components
print("x_reduced:")
print(x_reduced)