import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
n_samples = 1000
n_features = 5
X = np.random.randn(n_samples, n_features)
y = (X[:, 0] * 2 + X[:, 1] - X[:, 2] ** 2 > 0).astype(int)

train_ratio = 0.8
split_index = int(n_samples * train_ratio)

indices = np.random.permutation(n_samples)
train_idx, test_idx = indices[:split_index], indices[split_index:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

clf = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
print("Training XGBClassifier...")
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
preds_probs = clf.predict_proba(X_test)

accuracy = accuracy_score(y_test, preds)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, preds))