from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import numpy as np 
import matplotlib.pyplot as plt
X = np.array([
      [1, 4],
    [2, 5],
    [3, 6],
    [4, 7],
    [5, 8],
    [6, 7],
    [7, 8],
    [8, 6]
])

Y = np.array([0, 0, 0, 0, 1, 1, 1, 1]
)
model = DecisionTreeClassifier(
    criterion ="gini",
    max_depth =3,
    random_state = 42
)

model.fit(X, Y)

prediction = model.predict([[5, 7]])

if prediction[0] == 1:
    print("The student is predicted to PASS.")
else:
    print("The student is predicted to FAIL.")

plt.figure(figsize=(8, 6))

plot_tree(
    model,
    feature_names=["Hours studied", "Hours slept"],
    class_names=["Fail", "Pass"],
    filled=True
)
plt.show()