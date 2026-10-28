import pandas as pd 

from sklearn.model_selection import  train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 

data = {
    "age": [22, 25, 28, 35, 40, 45, 50, 55, 60, 65],
    "income": [20000, 25000, 30000, 40000, 45000,
               50000, 60000, 65000, 70000, 80000],
    "bought": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["age", "income"]]
Y = df["bought"]

X_train , X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,  random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print("prediction:", Y_pred)

print("accuracy:", accuracy_score(Y_test, Y_pred))