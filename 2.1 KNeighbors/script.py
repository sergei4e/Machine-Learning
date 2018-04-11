import pandas
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale

data = pandas.read_csv('wine.csv')

x = data.as_matrix(columns=[x for x in data if x != 'Class'])
y = data['Class']

neigh = KNeighborsClassifier(n_neighbors=3)
kf = KFold(len(data), n_folds=5, shuffle=True, random_state=42)

for i in range(1, 51):
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=i), scale(x), y, cv=kf)
    res = sum(scores) / 5.0
    print(i, format(res, '.2f'))

import pdb
pdb.set_trace()
