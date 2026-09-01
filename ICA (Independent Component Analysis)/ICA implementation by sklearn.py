import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

np.random.seed(42)

n_samples = 1000
t = np.linspace(0, 8, n_samples)

source_1 = np.sin(2 * t)
source_2 = np.sign(np.sin(3 * t))

S = np.c_[source_1, source_2]


A = np.array([
    [1.0, 0.5],
    [0.4, 1.0]
])

X = S @ A.T


ica = FastICA(
    n_components=2,
    random_state=42
)

S_recovered = ica.fit_transform(X)


plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(S[:, 0])
plt.title("Original Source 1")

plt.subplot(2, 1, 2)
plt.plot(S[:, 1])
plt.title("Original Source 2")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(X[:, 0])
plt.title("Mixed Signal 1")

plt.subplot(2, 1, 2)
plt.plot(X[:, 1])
plt.title("Mixed Signal 2")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(S_recovered[:, 0])
plt.title("Recovered Source 1")

plt.subplot(2, 1, 2)
plt.plot(S_recovered[:, 1])
plt.title("Recovered Source 2")

plt.tight_layout()
plt.show()

print("Original mixing matrix A:")
print(A)

print("Learned mixing matrix:")
print(ica.mixing_)

print("Learned unmixing matrix W:")
print(ica.components_)