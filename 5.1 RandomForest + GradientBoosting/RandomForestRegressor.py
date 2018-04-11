import pandas
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
import pdb

data = pandas.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

columns = [x for x in data.keys() if x != 'Rings']
x = data.as_matrix(columns=columns)
y = data['Rings'].values

clf = RandomForestRegressor(random_state=1)

grid = {'n_estimators': np.arange(1, 51)}

kf = KFold(y.size, n_folds=5, shuffle=True, random_state=1)
gs = GridSearchCV(clf, grid, scoring='r2', cv=kf)
gs.fit(x, y)

for s in gs.grid_scores_:
    if s.mean_validation_score > 0.52:
        print(s)
        break

pdb.set_trace()
