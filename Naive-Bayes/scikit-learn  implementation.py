from sklearn.naive_bayes import GaussianNB

X_train = [
    [150, 8],
    [170, 7],
    [140, 9],
    [130, 8],
    [220, 3],
    [250, 2],
    [230, 4],
    [240, 3]
]
y_train = [0, 0, 0, 0, 1, 1, 1, 1]

model = GaussianNB()
model.fit(X_train, y_train)


X_test = [
    [145, 8], 
    [235, 3]   
]
y_pred = model.predict(X_test)

print("Predictions:", y_pred)