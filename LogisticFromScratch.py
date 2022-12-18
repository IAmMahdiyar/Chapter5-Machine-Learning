import numpy as np 

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

x_train = np.array([[6, 7], [2, 4], [3, 6], [4, 7], [1, 6], [5, 2], [2, 0], [6, 3], [4, 1], [7, 2]])
y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1,1])

weights = train_logistic_regression(x_train, y_train, 1000, 0.1, fit_intercept=True)
predict(x_train, weights)
