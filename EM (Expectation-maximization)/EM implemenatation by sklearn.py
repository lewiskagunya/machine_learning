import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.mixture import GaussianMixture

np.random.seed(42)

X = np.vstack([
    np.random.randn(100, 2) + [2, 2],
    np.random.randn(100, 2) + [7, 7],
    np.random.randn(100, 2) + [2, 7]
])

model = GaussianMixture(n_components=3, 
                        random_state=42)
model.fit(X)
print("means:")
print(model.means_)
print("matrices covariances:")
print(model.covariances_)
print("weights")
print(model.weights_)

responsibilities = model.predict_proba(X)
print("responsibility shape:")
print(responsibilities.shape)
print("probabilities of gaussian 1:")
print(responsibilities[1])

labels = model.predict(X)
print("labels of 10 and 100:")
print(labels[:10])
print(labels[:100])

indices = np.where(labels[:100] == 1)[0]

print("indices:", indices)
print("points:", X[indices])
print("responsibilities:", responsibilities[indices])

plt.scatter(X[:,0],X[:,1], c=labels, cmap ='viridis', label ='clusters')
plt.title("EM representation")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.legend()
plt.show()