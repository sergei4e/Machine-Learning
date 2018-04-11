import pandas
from sklearn.svm import SVC

data = pandas.read_csv('svm-data.csv')

x = data.as_matrix(columns=['f1', 'f2'])
y = data['obj']

clf = SVC(kernel='linear', C=100000, random_state=241)
X = clf.fit(x, y)

print(x, y)

print(clf.support_)
