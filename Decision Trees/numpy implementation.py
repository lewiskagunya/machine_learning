
import numpy as np
X = np.array([
    [1, 4],
    [2, 5],
    [3, 6],
    [4, 7],
    [5, 8],
    [6, 7],
    [7, 8],
    [8, 6]
], dtype=float)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

class Decision_tree():
    def __init__(self, max_depth=3, mini_samples_split=2):
        self.max_depth = max_depth
        self.mini_samples_split = mini_samples_split
        self.tree = None

    def gini_impurity(self, y):
        if len(y) == 0:
            return 0
        counts = np.unique(y, return_counts=True)
        probabilities = counts/len(y)
        return 1 - np.sum(probabilities ** 2)
    def best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None

        parent_gini = self.gini(y)
        n_samples, n_features = X.shape
        for  feature in range(n_features):
            values = np.unique(X[:, feature])
            thresholds = (
                (values[:-1] + values[1:]) / 2
            )

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                y_left = y[left_mask]
                y_right = y[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                weighted_gini = (
                    len(y_left) / n_samples * self.gini(y_left)
                    + len(y_right) / n_samples * self.gini(y_right)
                )

                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain
    def majority_class(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    # Build the tree recursively
    def build_tree(self, X, y, depth=0):

        # Stop when the node is pure or too small
        if (
            len(np.unique(y)) == 1
            or depth >= self.max_depth
            or len(y) < self.min_samples_split
        ):
            return {
                "leaf": True,
                "prediction": self.majority_class(y)
            }

        feature, threshold, gain = self.best_split(X, y)

        # Stop if no useful split exists
        if feature is None or gain <= 0:
            return {
                "leaf": True,
                "prediction": self.majority_class(y)
            }

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self.build_tree(
                X[left_mask], y[left_mask], depth + 1
            ),
            "right": self.build_tree(
                X[right_mask], y[right_mask], depth + 1
            )
        }

    # Train the model
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
        return self

    # Predict one example
    def predict_one(self, x, node=None):
        if node is None:
            node = self.tree

        if node["leaf"]:
            return node["prediction"]

        feature = node["feature"]
        threshold = node["threshold"]

        if x[feature] <= threshold:
            return self.predict_one(x, node["left"])

        return self.predict_one(x, node["right"])

    # Predict multiple examples
    def predict(self, X):
        return np.array([
            self.predict_one(x) for x in X
        ])


# -----------------------------
# 3. Train and test the tree
# -----------------------------

model = Decision_tree(max_depth=3)
model.fit(X, y)

predictions = model.predict(X)

print("Actual labels:    ", y)
print("Predicted labels: ", predictions)
print("Training accuracy:",
      np.mean(predictions == y))
