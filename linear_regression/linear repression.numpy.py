import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. MODEL IMPLEMENTATION
# =====================================================================

class LinearRegressionScratch:
    """
    Linear Regression model built from scratch using NumPy.
    Optimized using Gradient Descent.
    """
    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 200):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        self.cost_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        """
        Train the model using Gradient Descent.
        
        Parameters:
            X: Input feature array of shape (m, n_features)
            y: Target array of shape (m, 1)
        """
        m, n_features = X.shape
        
        # Initialize parameters
        self.w = np.zeros((n_features, 1))
        self.b = 0.0
        self.cost_history = []

        for _ in range(self.n_iterations):
            # Forward pass: compute predictions
            y_pred = X @ self.w + self.b
            
            # Compute Mean Squared Error (Cost)
            cost = (1 / m) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)

            # Compute gradients
            dw = (2 / m) * (X.T @ (y_pred - y))
            db = (2 / m) * np.sum(y_pred - y)

            # Parameter updates
            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for given input array X."""
        return X @ self.w + self.b


# =====================================================================
# 2. DATA GENERATION & TRAINING
# =====================================================================

if __name__ == "__main__":
    # Reproducibility seed
    np.random.seed(42)

    # Synthetic Dataset: True relationship y = 3.0 * X + 4.0 + noise
    X = 2 * np.random.rand(100, 1)
    y = 3.0 * X + 4.0 + np.random.randn(100, 1) * 0.5

    # Train Model
    model = LinearRegressionScratch(learning_rate=0.1, n_iterations=200)
    model.fit(X, y)

    # Print Learned Parameters vs Target
    print("=" * 45)
    print("  LINEAR REGRESSION GRADIENT DESCENT RESULTS")
    print("=" * 45)
    print(f"Learned Weight (w): {model.w[0][0]:.4f} | Target: ~3.0000")
    print(f"Learned Bias   (b): {model.b:.4f} | Target: ~4.0000")
    print(f"Final Loss   (MSE): {model.cost_history[-1]:.4f}")
    print("=" * 45)

    # Generate smooth test predictions for plotting
    X_line = np.linspace(0, 2, 100).reshape(-1, 1)
    y_line = model.predict(X_line)

    # =====================================================================
    # 3. VISUALIZATION
    # =====================================================================
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Regression Fit
    axes[0].scatter(X, y, color='#1f77b4', alpha=0.7, edgecolors='k', label='Observed Data')
    axes[0].plot(X_line, y_line, color='#d62728', linewidth=2.5, label='Learned Line')
    axes[0].set_title('Linear Regression Model Fit', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Feature Space (X)', fontsize=10)
    axes[0].set_ylabel('Target Value (y)', fontsize=10)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Loss Curve
    axes[1].plot(range(1, model.n_iterations + 1), model.cost_history, color='#2ca02c', linewidth=2)
    axes[1].set_title('Loss Reduction (MSE vs Iterations)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Iteration', fontsize=10)
    axes[1].set_ylabel('Mean Squared Error', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()