import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
num_samples = 200

# Class 0: Center around (-1.5, -1.5)
class_0 = np.random.randn(num_samples // 2, 2) - 1.5
y0 = np.zeros(num_samples // 2)

# Class 1: Center around (1.5, 1.5)
class_1 = np.random.randn(num_samples // 2, 2) + 1.5
y1 = np.ones(num_samples // 2)

# Combine into single dataset
X = np.vstack((class_0, class_1))
y = np.hstack((y0, y1))


# ==========================================
# 2. LOGISTIC REGRESSION FROM SCRATCH
# ==========================================
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Prevent exp overflow
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_hat):
    m = len(y)
    # Clip probabilities to prevent log(0)
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return - (1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


learning_rate = 0.1
iterations = 500


num_samples, num_features = X.shape
w = np.zeros(num_features)
b = 0.0


loss_history = []


for i in range(iterations):
 
    linear_model = np.dot(X, w) + b
    y_hat = sigmoid(linear_model)

    loss = compute_loss(y, y_hat)
    loss_history.append(loss)
   
    dw = (1 / num_samples) * np.dot(X.T, (y_hat - y))
    db = (1 / num_samples) * np.sum(y_hat - y)
    
    # Update Parameters
    w -= learning_rate * dw
    b -= learning_rate * db


predictions = (y_hat >= 0.5).astype(int)
accuracy = np.mean(predictions == y) * 100
print(f"Final Loss: {loss_history[-1]:.4f}")
print(f"Training Accuracy: {accuracy:.2f}%")


# ==========================================
# 3. VISUALIZATION (Matplotlib)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))


axes[0].plot(loss_history, color='blue', linewidth=2)
axes[0].set_title("Loss Reduction Over Iterations")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Binary Cross-Entropy Loss")
axes[0].grid(True)


axes[1].scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0', alpha=0.7)
axes[1].scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1', alpha=0.7)


x1_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
x2_vals = -(w[0] * x1_vals + b) / w[1]

axes[1].plot(x1_vals, x2_vals, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
axes[1].set_title("Learned Decision Boundary")
axes[1].set_xlabel("Feature 1 ($x_1$)")
axes[1].set_ylabel("Feature 2 ($x_2$)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()