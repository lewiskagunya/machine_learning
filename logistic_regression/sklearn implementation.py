import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

x = np.array([[1, 2], [2, 3], [3, 1], [4, 5], [5, 4]])
y = np.array([0, 0, 0, 1, 1])

model = LogisticRegression()
model.fit(x, y)

new_x  = np.array([[2, 2], [3, 4], [5, 5]])
print("Predicted classes for new data points:", model.predict(new_x))

w = model.coef_[0]
b = model.intercept_[0]

print("slope:", w)
print("intercept:", b)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', label='Training Data')
plt.scatter(new_x[:, 0], new_x[:, 1], c='green', marker='o', label='New Data Points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression')
plt.legend()
plt.show()