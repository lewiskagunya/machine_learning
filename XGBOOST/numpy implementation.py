
import numpy as np

# ==========================================
# 1. CREATE THE DATASET
# ==========================================

X = np.array([
    [50],
    [60],
    [70],
    [80],
    [90],
    [100]
], dtype=float)

y = np.array([100, 120, 140, 160, 180, 200], dtype=float)


# ==========================================
# 2. CALCULATE SPLIT GAIN
# ==========================================

def calculate_gain(
    G_left, H_left,
    G_right, H_right,
    G_parent, H_parent,
    lambda_reg, gamma
):
    left_score = G_left**2 / (H_left + lambda_reg)
    right_score = G_right**2 / (H_right + lambda_reg)
    parent_score = G_parent**2 / (H_parent + lambda_reg)

    gain = 0.5 * (
        left_score + right_score - parent_score
    ) - gamma

    return gain


# ==========================================
# 3. TRAIN ONE TREE WITH ONE SPLIT
# ==========================================

def fit_stump(X, gradient, hessian, lambda_reg, gamma):
    feature = X[:, 0]

    thresholds = (
        feature[:-1] + feature[1:]
    ) / 2

    G_parent = np.sum(gradient)
    H_parent = np.sum(hessian)

    best_gain = 0.0
    best_threshold = None
    best_left_score = None
    best_right_score = None

    for threshold in thresholds:
        left_mask = feature <= threshold
        right_mask = ~left_mask

        G_left = np.sum(gradient[left_mask])
        H_left = np.sum(hessian[left_mask])

        G_right = np.sum(gradient[right_mask])
        H_right = np.sum(hessian[right_mask])

        gain = calculate_gain(
            G_left, H_left,
            G_right, H_right,
            G_parent, H_parent,
            lambda_reg, gamma
        )

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

            best_left_score = (
                -G_left / (H_left + lambda_reg)
            )

            best_right_score = (
                -G_right / (H_right + lambda_reg)
            )

    return (
        best_threshold,
        best_left_score,
        best_right_score,
        best_gain
    )


# ==========================================
# 4. MODEL SETTINGS
# ==========================================

learning_rate = 0.1
n_estimators = 5
lambda_reg = 1.0
gamma = 0.0

# Initial prediction: mean target value
prediction = np.full(len(y), np.mean(y))

# Store each tree's information
trees = []


# ==========================================
# 5. BOOSTING TRAINING LOOP
# ==========================================

for iteration in range(n_estimators):

    # Gradients and Hessians for squared-error loss
    gradient = prediction - y
    hessian = np.ones(len(y))

    # Train a new tree using the current gradients
    threshold, left_score, right_score, gain = fit_stump(
        X,
        gradient,
        hessian,
        lambda_reg,
        gamma
    )

    # Stop if no positive-gain split is found
    if threshold is None:
        print("No beneficial split found.")
        break

    # Save the tree
    trees.append((threshold, left_score, right_score))

    # Calculate this tree's predictions
    tree_output = np.where(
        X[:, 0] <= threshold,
        left_score,
        right_score
    )

    # Update the existing predictions
    prediction += learning_rate * tree_output

    # Evaluate training loss
    residuals = y - prediction
    loss = 0.5 * np.sum(residuals**2)

    print(
        f"Round {iteration + 1}: "
        f"threshold={threshold:.1f}, "
        f"gain={gain:.2f}, "
        f"loss={loss:.2f}"
    )


# ==========================================
# 6. DISPLAY FINAL PREDICTIONS
# ==========================================

print("\nActual prices:")
print(y)

print("\nPredicted prices:")
print(np.round(prediction, 2))

print("\nFinal training loss:")
print(round(0.5 * np.sum((y - prediction)**2), 2))


# ==========================================
# 7. PREDICT FOR NEW HOUSE SIZES
# ==========================================

def predict(X_new, initial_prediction, trees, learning_rate):
    predictions = np.full(
        len(X_new),
        initial_prediction,
        dtype=float
    )

    for threshold, left_score, right_score in trees:
        tree_output = np.where(
            X_new[:, 0] <= threshold,
            left_score,
            right_score
        )

        predictions += learning_rate * tree_output

    return predictions


new_houses = np.array([
    [55],
    [75],
    [95]
], dtype=float)

new_predictions = predict(
    new_houses,
    np.mean(y),
    trees,
    learning_rate
)

print("\nNew house sizes:")
print(new_houses[:, 0])

print("\nPredicted prices:")
print(np.round(new_predictions, 2))
