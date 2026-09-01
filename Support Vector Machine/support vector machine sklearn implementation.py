import numpy as np 
from sklearn.svm import SVC 
import matplotlib.pyplot as plt 

x = np.array([[1,2],[2,3],[3,1],[5,4],[6,5],[7,8]])

y = np.array([0,0,0,1,1,1]) 
model = SVC(kernel='linear')

model.fit(x,y)
predictions = model.predict(x)

print("Predictions:", predictions)
print("weights:", model.coef_)
print("intercept:", model.intercept_)
print("support vectors:", model.support_vectors_)
print("support indices:", model.support_)
print("dual coefficients:", model.dual_coef_)

plt.scatter(x[:,0], x[:,1], c=y, cmap='coolwarm', label='Training Data')
w = model.coef_[0]
b = model.intercept_[0]
x_plot = np.linspace(0, 8, 100)
y_plot = -(w[0]/w[1])*x_plot - b/w[1]
plt.plot(x_plot, y_plot, color='black', label='Decision Boundary')

plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1],
             s=100, facecolors='none', edgecolors='k', label='Support Vectors')
plt.legend()
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.show()