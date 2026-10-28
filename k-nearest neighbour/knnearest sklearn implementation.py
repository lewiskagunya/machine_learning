from sklearn.neighbors import KNeighborsRegressor
import numpy as np 

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

y = np.array([
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80
])

model  = KNeighborsRegressor(n_neighbors = 3)
model.fit(X,y)
x_new = np.array([[4.5]])

prediction = model.predict(x_new)
print(prediction)