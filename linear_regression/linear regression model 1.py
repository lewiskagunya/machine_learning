import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1000], [2000],[3200],[3500],[4000],[4500]], dtype=float)

y = np.array([100,200,360,400,450 , 500], dtype=float)

m = len(x)
learning_rate = 1e-8
theta0 = 0.0
theta1 = 0.0

for i in range(1000):
    y_pred = theta0 + theta1 * x

    error = y_pred - y
    cost = (1/(2*m)) * np.sum(error ** 2)
    d_theta0 = (1/m) * np.sum(error)
    d_theta1 = (1/m) * np.sum(error * x)

    theta0 = theta0 - learning_rate * d_theta0
    theta1 = theta1 - learning_rate * d_theta1

print("Theta0:", theta0)
print("Theta1:", theta1)
print("Final Cost:", cost)
plt.scatter(x, y, color="blue", label="Training Data")
plt.plot(x, theta0 + theta1 * x, color="red", label="Regression Line")

plt.xlabel("House Size")
plt.ylabel("Price")
plt.title("Linear Regression From Scratch")

plt.legend()
plt.show()



