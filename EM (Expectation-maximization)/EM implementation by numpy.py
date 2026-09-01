import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CREATE THE DATA
# ============================================================

np.random.seed(42)

X = np.vstack([
    np.random.randn(100, 2) + [2, 2],
    np.random.randn(100, 2) + [7, 7],
    np.random.randn(100, 2) + [2, 7]
])


# ============================================================
# 2. INITIALIZATION
# ============================================================

K = 3
n, d = X.shape

# Randomly choose K data points as initial means
means = X[np.random.choice(n, K, replace=False)]

# Start each covariance matrix as the identity matrix
covariances = np.array([
    np.eye(d)
    for _ in range(K)
])

# Start with equal mixture weights
weights = np.ones(K) / K


# ============================================================
# 3. MULTIVARIATE GAUSSIAN PDF
# ============================================================

def gaussian_pdf(X, mean, covariance):

    d = X.shape[1]

    covariance_inv = np.linalg.inv(covariance)
    covariance_det = np.linalg.det(covariance)

    diff = X - mean

    exponent = -0.5 * np.sum(
        (diff @ covariance_inv) * diff,
        axis=1
    )

    coefficient = 1 / (
        (2 * np.pi) ** (d / 2)
        * np.sqrt(covariance_det)
    )

    return coefficient * np.exp(exponent)


# ============================================================
# 4. E-STEP
# ============================================================

def e_step(X, means, covariances, weights):

    n = len(X)

    responsibilities = np.zeros((n, K))

    for j in range(K):

        pdf = gaussian_pdf(
            X,
            means[j],
            covariances[j]
        )

        responsibilities[:, j] = weights[j] * pdf

    # Normalize each row so that responsibilities sum to 1
    responsibilities /= responsibilities.sum(
        axis=1,
        keepdims=True
    )

    return responsibilities


# ============================================================
# 5. M-STEP
# ============================================================

def m_step(X, responsibilities):

    n, d = X.shape

    # Effective number of points belonging to each Gaussian
    Nk = responsibilities.sum(axis=0)

    # --------------------------------------------------------
    # Update means
    # --------------------------------------------------------

    means = np.zeros((K, d))

    for j in range(K):

        means[j] = (
            responsibilities[:, j][:, np.newaxis] * X
        ).sum(axis=0) / Nk[j]

    # --------------------------------------------------------
    # Update covariance matrices
    # --------------------------------------------------------

    covariances = np.zeros((K, d, d))

    for j in range(K):

        diff = X - means[j]

        covariances[j] = (
            responsibilities[:, j][:, np.newaxis] * diff
        ).T @ diff / Nk[j]

    # --------------------------------------------------------
    # Update mixture weights
    # --------------------------------------------------------

    weights = Nk / n

    return means, covariances, weights


# ============================================================
# 6. LOG-LIKELIHOOD
# ============================================================

def log_likelihood(X, means, covariances, weights):

    total_probability = np.zeros(len(X))

    for j in range(K):

        total_probability += (
            weights[j]
            * gaussian_pdf(
                X,
                means[j],
                covariances[j]
            )
        )

    return np.sum(np.log(total_probability))


# ============================================================
# 7. EM ALGORITHM
# ============================================================

max_iterations = 100
tolerance = 1e-6

previous_log_likelihood = None

for iteration in range(max_iterations):

    # -------------------------
    # E-step
    # -------------------------

    responsibilities = e_step(
        X,
        means,
        covariances,
        weights
    )

    # -------------------------
    # M-step
    # -------------------------

    means, covariances, weights = m_step(
        X,
        responsibilities
    )

    # -------------------------
    # Calculate log-likelihood
    # -------------------------

    current_log_likelihood = log_likelihood(
        X,
        means,
        covariances,
        weights
    )

    print(
        f"Iteration {iteration + 1}: "
        f"log-likelihood = {current_log_likelihood:.6f}"
    )

    # -------------------------
    # Check convergence
    # -------------------------

    if previous_log_likelihood is not None:

        if abs(
            current_log_likelihood
            - previous_log_likelihood
        ) < tolerance:

            print("EM converged.")

            break

    previous_log_likelihood = current_log_likelihood


# ============================================================
# 8. FINAL CLUSTER LABELS
# ============================================================

labels = np.argmax(
    responsibilities,
    axis=1
)


# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\nFinal means:")
print(means)

print("\nFinal covariance matrices:")
print(covariances)

print("\nFinal weights:")
print(weights)

print("\nResponsibility shape:")
print(responsibilities.shape)

print("\nFirst 10 labels:")
print(labels[:10])


# ============================================================
# 10. VISUALIZE
# ============================================================

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis"
)

plt.scatter(
    means[:, 0],
    means[:, 1],
    marker="X",
    s=200,
    label="Gaussian means"
)

plt.title("Gaussian Mixture Model using EM")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.show()