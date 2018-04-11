import pandas
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
import pdb

import matplotlib.pyplot as plt
plt.figure()


data = pandas.read_csv('gbm-data.csv')

columns = [x for x in data.keys() if x != 'Activity']
x = data.as_matrix(columns=columns)
y = data['Activity'].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=241)

'''
loss_train = {}
loss_test = {}

for k in [1, 0.5, 0.3, 0.2, 0.1]:
    loss_train[k] = list()
    loss_test[k] = list()


for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
    clf1 = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=learning_rate)
    clf1.fit(X_train, y_train)

    for i, y_decision in enumerate(clf1.staged_decision_function(X_train)):
        y_pred = 1.0 / (1.0 + np.exp(-y_decision))
        lll = log_loss(y_train, y_pred)
        loss_train[learning_rate].append(lll)

    for j, y_decision in enumerate(clf1.staged_decision_function(X_test)):
        y_pred = 1.0 / (1.0 + np.exp(-y_decision))
        lll = log_loss(y_test, y_pred)
        loss_test[learning_rate].append(lll)

colors = []

for i in loss_test:
    plt.plot(loss_test[i], linewidth=1)
    plt.plot(loss_train[i], linewidth=1)
    colors.append('test'+str(i))
    colors.append('train'+str(i))

plt.legend(colors)

clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=0.2)
clf.fit(X_train, y_train)

loss_tr = {}
for i, y_decision in enumerate(clf.staged_decision_function(X_test)):
    y_pred = 1.0 / (1.0 + np.exp(-y_decision))
    lll = log_loss(y_test, y_pred)
    loss_tr[i] = lll

print(min(loss_tr.items(), key=(lambda x: x[1])))

plt.show()

'''

clf = RandomForestClassifier(random_state=241, n_estimators=36)
clf.fit(X_train, y_train)
pred = clf.predict_proba(X_test)
l = log_loss(y_test, pred)

print(l)

pdb.set_trace()
