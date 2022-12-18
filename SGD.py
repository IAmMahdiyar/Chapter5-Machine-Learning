from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import time

n_rows= 100000

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

start_time = time.time()
sgd = SGDClassifier(max_iter=1000, learning_rate='constant', loss='log_loss', eta0 = 0.01, penalty=None)
sgd.fit(x_train_enc, y_train)
print(time.time() - start_time, "second")

pred = sgd.predict_proba(x_test_enc)[:, 1]
print("Score:", roc_auc_score(y_test, pred))