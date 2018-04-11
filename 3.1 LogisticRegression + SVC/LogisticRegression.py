import pandas
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

data = pandas.read_csv('data-logistic.csv')

x = data.as_matrix(columns=['f1', 'f2'])
y = data['obj']

clf = LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=10.0,
                         fit_intercept=True, intercept_scaling=0.1,
                         class_weight=None, random_state=None, solver='liblinear',
                         max_iter=10000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
clf.fit(x, y)

koef = clf.decision_function(x)

print(clf.coef_)

# инициализация констант
k = 0.1
C = 10.0
epsilon = 1e-5
max_it = 10000

# инициализация счётчика итераций
it = 0

# первое приближение коэффициентов
w1 = 0.0
w2 = 0.0

# Sigmoidnaya a(x) = 1 / (1 + exp(-w1 x1 - w2 x2))


def get_w(W1, W2):
    l = len(y)
    ww1, ww2 = 0,0
    for i in range(len(y)):
        ww1 = ww1 + k * 1/l * y[i]*x[i][0]*(1 - 1/(1 + math.exp(-y[i] * (W1*x[i][0] + W2*x[i][1]))))
        ww2 = ww2 + k * 1/l * y[i]*x[i][1]*(1 - 1/(1 + math.exp(-y[i] * (W1*x[i][0] + W2*x[i][1]))))

    w1 = W1 + ww1  # - k*C*W1
    w2 = W2 + ww2  # - k*C*W2

    return w1, w2

# основной цикл
while it <= max_it:
    w1_new, w2_new = get_w(w1, w2)
    if abs(w1 - w1_new) < epsilon and abs(w2 - w2_new) < epsilon:
        break
    w1 = w1_new
    w2 = w2_new
    it = it + 1

print(it)
print(w1, w2)

res = []

for i in x:
    res.append(1 / (1 + math.exp(-w1*i[0] - w2*i[1])))

print(roc_auc_score(y, res))
