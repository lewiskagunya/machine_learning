import numpy as np 
x = np.array([
    [1, 2],
    [2, 3],
    [3, 1],
    [4, 5],
    [5, 4],
    [6, 6]
], dtype=float)

y = np.array([-1, -1, -1, 1, 1, 1])

w = np.zeros(x.shape[1])
b = 0.0 

learning_rate = 0.01
epochs = 1000
lambda_ = 0.01

for epoch in range(epochs):
    for i in range(len(x)):
        condition = y[i] * (np.dot(x[i], w) + b)
        if condition >= 1:
            dw = 2 * lambda_ * w
        else:
            dw = 2 * lambda_ * w - y[i] * x[i]
            db = -y[i]

        w -= learning_rate * dw
        b -= learning_rate * db

print("Weights:", w)
print("Bias:", b)
