import numpy as np 
from sklearn.neural_network import MLPClassifier

x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])
model = MLPClassifier( hidden_layer_sizes=(4,),
    activation="relu",
    solver="lbfgs",
    max_iter=1000,
    random_state=42)
model.fit(x, y)

predictions = model.predict(x)

print("predictions:",predictions)
print("Actual:", y)


for i, weights in enumerate(model.coefs_):
    print(f"\nLayer {i + 1}:")
    print(weights)

for i, bias in enumerate(model.intercepts_):
    print(f"\nLayer {i + 1}:")
    print(bias)

print("\nNetwork structure:")
print("Input features:", X.shape[1])
print("Hidden layers:", model.hidden_layer_sizes)
print("Output classes:", len(model.classes_))