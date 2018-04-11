import numpy as np
from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale

data = datasets.load_boston()

x = scale(data['data'])
y = data['target']

kf = KFold(len(x), n_folds=5, shuffle=True, random_state=42)

res = {}

for i in np.linspace(1.0, 10.0, num=200):
    scores = cross_val_score(KNeighborsRegressor(n_neighbors=5, weights='distance', p=i, metric='minkowski'),
                             x, y, scoring='mean_squared_error', cv=kf)
    res[str(format(i, '.2f'))] = format(sum(scores) / 5.0, '.2f')

val = min(res.items(), key=(lambda x: x[1]))
print(val)


