import numpy as np 

x = np.array([[1, 2],
    [2, 3],
    [3, 1],
    [4, 5],
    [5, 4]])
y = np.array([ [0],
    [0],
    [0],
    [1],
    [1]])

np.random.seed(42)

W1 = np.random.randn(2, 3)
b1 = np.zeros((1, 3))

W2 = np.random.randn(3, 1)
b2 = np.zeros((1, 1))

def sigmoid(z):
    return 1/( 1 +np.exp(-z))
z1 = x @ W1 + b1
A1 = sigmoid(z1)
z2 = A1 @ W2 + b2
y_hat =sigmoid(z2)

print("Predictions:")
print(y_hat)

m = len(y)

loss = -(1/m) * np.sum(
    y * np.log(y_hat) +
    (1-y) * np.log(1-y_hat)
)
print("Loss:", loss)
dZ2 = y_hat - y
dW2 = (1/m) * (A1.T @ dZ2)
db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

dA1 = dZ2 @ W2.T

dZ1 = dA1 * A1 * (1-A1)
dW1 = (1/m) * (x.T @ dZ1)
db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

learning_rate = 0.1

W2 -= learning_rate * dW2
b2 -= learning_rate * db2

W1 -= learning_rate * dW1
b1 -= learning_rate * db1