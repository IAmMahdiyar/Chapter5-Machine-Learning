from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import time

n_rows= 100000 * 11

df = pd.read_csv("train.csv", nrows=n_rows)

print("Loaded")

Y = df["click"].values
X = df.drop(["click", "id", "hour", "device_id", "device_ip"], axis=1).values

n_click = (X==1).sum()
n_not_click = (X == 0).sum()
print(n_click / (n_click + n_not_click) * 100, "Percent Have Clicked")

n_train = n_rows * 10
x_train = X[ :n_train]
y_train = Y[ :n_train]
x_test = X[int(n_train / 11): ]
y_test = Y[int(n_train / 11): ]

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(x_train)
x_test_enc = enc.transform(x_test)

start_time = time.time()
sgd = SGDClassifier(max_iter=1000, learning_rate='constant', loss='log_loss', eta0 = 0.01, penalty=None)
for i in range(10):
    x = x_train[i*100000:(i+1)*100000]
    y = y_train[i*100000:(i+1)*100000]
    x = enc.transform(x)
    sgd.partial_fit(x.toarray(), y, classes=[0, 1])
    print("Fitted NO.", str(i + 1))

print(time.time() - start_time, "second")

pred = sgd.predict_proba(x_test_enc)[:, 1]
print("Score:", roc_auc_score(y_test, pred))