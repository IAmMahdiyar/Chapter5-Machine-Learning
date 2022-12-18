import numpy as np
import timeit
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

n_rows=  10000
df = pd.read_csv("train.csv", nrows=n_rows)

print("Loaded")

Y = df["click"].values
X = df.drop(["click", "id", "hour", "device_id", "device_ip"], axis=1).values

n_click = (X==1).sum()
n_not_click = (X == 0).sum()
print(n_click / (n_click + n_not_click) * 100, "Percent Have Clicked")

n_train = int(n_rows * 0.9)
x_train = X[ :n_train]
y_train = Y[ :n_train]
x_test = X[n_train: ]
y_test = Y[n_train: ]

enc = OneHotEncoder(handle_unknown='ignore')
x_train_enc = enc.fit_transform(x_train)
x_test_enc = enc.transform(x_test)

def sigmoid(input):
    return 1.0 / (1 + np.exp(-input))

def compute_prediction(X, weights):
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    return predictions

def update_weights_gd(X_train, y_train, weights, learning_rate):
    predictions = compute_prediction(X_train, weights)
    weights_delta = np.dot(X_train.T, y_train - predictions)
    m = y_train.shape[0]
    weights += learning_rate / float(m) * weights_delta
    return weights

def compute_cost(X, y, weights):
    predictions = compute_prediction(X, weights)
    cost = np.mean(-y * np.log(predictions)
    - (1 - y) * np.log(1 - predictions))
    return cost

def train_logistic_regression(X_train, y_train, max_iter, learning_rate, fit_intercept=False):
    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
        weights = np.zeros(X_train.shape[1])
        
        for iteration in range(max_iter):
            
            weights = update_weights_gd(X_train, y_train, weights, learning_rate)

            if iteration % 100 == 0:
                print(compute_cost(X_train, y_train, weights))
        return weights

def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
    return compute_prediction(X, weights)

weights = train_logistic_regression(x_train_enc.toarray(), y_train, 10000, 0.01, fit_intercept=True)
pred = predict(x_test_enc.toarray(), weights)
print(roc_auc_score(y_test, pred))