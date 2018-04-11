import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

train = pandas.read_csv('perceptron-train.csv')
test = pandas.read_csv('perceptron-test.csv')

x = train.as_matrix(columns=['f1', 'f2'])
y = train['obj']

xt = test.as_matrix(columns=['f1', 'f2'])
yt = test['obj']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x)
X_test_scaled = scaler.transform(xt)

clf = Perceptron(random_state=241)
clf.fit(x, y)
predictions = clf.predict(xt)
res = accuracy_score(yt, predictions)

clf2 = Perceptron(random_state=241)
clf2.fit(X_train_scaled, y)
predictions2 = clf2.predict(X_test_scaled)
res2 = accuracy_score(yt, predictions2)

print(res)
print(res2)
print(res2-res)
