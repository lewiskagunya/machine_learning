import numpy as np 

def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

y = np.array([0, 0, 1, 1,1,2,2])
print("gini impurity:", gini_impurity(y))
print("gini impurity for new node:", gini_impurity(np.array([0,1,1,2])))

def gini_weighted(y_left, y_right):
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0
    left_weight = n_left / n_total
    right_weight = n_right / n_total 

    return(
        left_weight * gini_impurity(y_left) + right_weight * gini_impurity(y_right)
    )
def best_split(x, y):
    best_feature = None
    best_threshold = None 
    best_impurity = float("inf")
    n_features = x.shape[1]

    for feature in range(n_features):
        values = np.unique(x[:, feature])
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2
            left_mask = x[:, feature] <= threshold
            right_mask = x[:, feature] >threshold
            y_left = y[left_mask]
            y_right = y[right_mask]
            impurity = gini_weighted(y_left, y_right)
            if impurity < best_impurity:
                best_impurity = impurity
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])
def majority_class(y):
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmax(counts)]

def build_tree(x, y, depth=0, max_depth=3, min_samples_split=2):
    prediction = majority_class(y)
    if len(np.unique(y)) == 1:
        return {"prediction": prediction}
    if depth >= max_depth or len(y) < min_samples_split:
        return {"prediction": prediction}
    feature, threshold, impurity= best_split(X, y)

    if feature is None:
        return {"prediction": prediction}

    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    if not np.any(left_mask) or not np.any(right_mask):
        return {"prediction": prediction}
    left_child = build_tree(
        X[left_mask], y[left_mask],
        depth + 1, max_depth, min_samples_split
    )

    right_child = build_tree(
        X[right_mask], y[right_mask],
        depth + 1, max_depth, min_samples_split
    )

    return {
        "feature": feature,
        "threshold": threshold,
        "left": left_child,
        "right": right_child,
        "prediction": prediction
    }
def predict_one(tree, x):
    if "feature" not in tree:
        return tree["prediction"]
    feature = tree["feature"]
    threshold = tree["threshold"]
    if x[feature] <= threshold:
        return predict_one(tree["left"], x)
    else:
        return predict_one(tree["right"], x)

def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])

y = np.array([0, 0, 0, 1, 1, 1])

feature, threshold = best_split(X, y)

print("Best feature index:", feature)
print("Best threshold:", threshold)