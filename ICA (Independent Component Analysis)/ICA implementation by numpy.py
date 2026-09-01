import numpy as np


# --------------------------------------------------
# 1. Create independent source signals
# --------------------------------------------------

np.random.seed(42)

n_samples = 1000

t = np.linspace(0, 8, n_samples)

source_1 = np.sin(2 * t)
source_2 = np.sign(np.sin(3 * t))

S = np.c_[source_1, source_2]


# --------------------------------------------------
# 2. Mix the sources
# --------------------------------------------------

A = np.array([
    [1.0, 0.5],
    [0.4, 1.0]
])

X = S @ A.T


# --------------------------------------------------
# 3. Center the mixed data
# --------------------------------------------------

mean = np.mean(X, axis=0)

X_centered = X - mean


# --------------------------------------------------
# 4. Initialize the unmixing matrix W
# --------------------------------------------------

W = np.random.randn(2, 2)


# --------------------------------------------------
# 5. Sigmoid function
# --------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --------------------------------------------------
# 6. ICA parameters
# --------------------------------------------------

learning_rate = 0.01
n_iterations = 1000

m = X_centered.shape[0]


# --------------------------------------------------
# 7. Learn W
# --------------------------------------------------

for i in range(n_iterations):

    # Estimate the independent sources
    Y = X_centered @ W.T

    # Apply the sigmoid non-linearity
    G = sigmoid(Y)

    # Calculate the update for W
    dW = (
        np.eye(2)
        - (G.T @ X_centered @ W.T) / m
    ) @ W

    # Update W
    W += learning_rate * dW


# --------------------------------------------------
# 8. Recover the independent sources
# --------------------------------------------------

S_recovered = X_centered @ W.T


# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("Original source shape:")
print(S.shape)

print("\nMixed data shape:")
print(X.shape)

print("\nLearned unmixing matrix W:")
print(W)

print("\nFirst 5 original samples:")
print(S[:5])

print("\nFirst 5 recovered samples:")
print(S_recovered[:5])