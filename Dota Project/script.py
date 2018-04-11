import pandas
import time
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pdb

# Загрузка данных из файла

train = pandas.read_csv('features.csv', index_col='match_id')
test = pandas.read_csv('features_test.csv', index_col='match_id')

# Определяем какие признаки заполнены не полностью

print('Не полностью заполнены:')

for i in train.keys():
    if train[i].size != train[i].count():
        print(i)

print('\n')
# Заменяем пустые значения на 0

for i in train.keys():
    train[i] = train[i].fillna(0)

# Определяем какие признаки есть в обучающем файле но нет в тестовом
print('То что нужно исключить из обучения. Признаки из будущего.')

exclude = [x for x in train.keys() if x not in test.keys()]
print(exclude)

print('\n')

# exclude = ['duration', 'radiant_win', 'tower_status_radiant',
#            'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire']

# Формируем X_train, X_test, Y_train, Y_test

X_train = train.as_matrix(columns=[x for x in train.keys() if x not in exclude])
y_train = train['radiant_win']

# Так как в тестовых данных нет правильных ответов - валидацию на них делать нельзя
# Тестовой выборкой будет часть обучающей

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.5)

# Первая часть: Градиентный бустинг

# Определяем оптимальное количество деревьев
# n_jobs=4 - распараллеливание рассчета на все ядра процессора

clf = GradientBoostingClassifier()
grid = {'n_estimators': [5, 10, 20, 30, 50]}
kf = KFold(y_train.size, n_folds=5, shuffle=True)
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=kf, n_jobs=4)
gs.fit(X_train, y_train)

print('Качество классификации Градиентного бустинга при различном количестве деревьев')
print(gs.grid_scores_)
print('Лучшее: ', max([(x.mean_validation_score, x.parameters) for x in gs.grid_scores_], key=lambda x: x[0]))
print('\n')

# Проводим кросс валидацию по 5 блокам, для 30 деревьев

t1 = time.time()
kf = KFold(y_train.size, n_folds=5, shuffle=True)
clf = GradientBoostingClassifier(n_estimators=30)
score = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=kf, n_jobs=4)
t2 = time.time()

print('Для 30 деревьев качество на кросс валидации по 5 блокам')
print(score)
print('Качество {}'.format(sum(score)/float(len(score))))
print('Время обработки Градиентного Бустинга на 30 деревьях {}'.format(t2-t1))
print('\n')
# Вторая часть: Логистическая Регрессия

# Проводим нормализацию значений, это улучшит качество

X_train = StandardScaler().fit_transform(X_train)

# Подбираем параметр 'C'

grid = {'C': np.power(10.0, np.arange(-5, 6))}
clf = LogisticRegression()
kf = KFold(y_train.size, n_folds=5, shuffle=True)
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=kf, n_jobs=4)
gs.fit(X_train, y_train)

best = max([(x.mean_validation_score, x.parameters) for x in gs.grid_scores_], key=lambda x: x[0])

print('Качество классификации Логистической Регрессии при различном параметре С')
print(gs.grid_scores_)
print('Лучшее: ', best)
print('\n')

# Вычисляем качество Логистической Регрессии с лучшим параметром С

t1 = time.time()
clf = LogisticRegression(**best[1])
kf = KFold(y_train.size, n_folds=5, shuffle=True)
score = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=kf, n_jobs=4)
t2 = time.time()
print('Для {} качество на кросс валидации по 5 блокам'.format(best[1]))
print(score)
print('Качество {}'.format(sum(score)/float(len(score))))
print('Время обработки Логистической Регрессии {}'.format(t2-t1))
print('\n')

#  Убираем категориальные признаки
#  lobby_type и r1_hero, r2_hero, ..., r5_hero, d1_hero, d2_hero, ..., d5_hero

categorial = ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
              'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']

x = train.as_matrix(columns=[x for x in train.keys() if x not in exclude and x not in categorial])
y = train['radiant_win']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
X_train = StandardScaler().fit_transform(X_train)

t1 = time.time()
clf = LogisticRegression(**best[1])
kf = KFold(y_train.size, n_folds=5, shuffle=True)
score = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=kf, n_jobs=4)
t2 = time.time()

print('Если уберем категориальные признаки из выборки, получим:')
print(score)
print('Качество {}'.format(sum(score)/float(len(score))))
print('Время обработки Логистической Регрессии {}'.format(t2-t1))
print('\n')

# Подключаем категориальные признаки обратно, но уже преобразованные

heroes = {}
N = len(train['r1_hero'].unique())
for n, k in enumerate(train['r1_hero'].unique()):
    heroes[k] = n

X_pick = np.zeros((train.shape[0], N))
for i, match_id in enumerate(train.index):
    for p in range(1, 6):
        X_pick[i, heroes[train.ix[match_id, 'r{}_hero'.format(p)]]] = 1
        X_pick[i, heroes[train.ix[match_id, 'd{}_hero'.format(p)]]] = -1

x = np.hstack((x, X_pick))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
X_train = StandardScaler().fit_transform(X_train)

t1 = time.time()
clf = LogisticRegression(**best[1])
kf = KFold(y_train.size, n_folds=5, shuffle=True)
score = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=kf, n_jobs=4)
t2 = time.time()

print('Если подключаем преобразованные категориальные признаки обратно, получим:')
print(score)
print('Качество {}'.format(sum(score)/float(len(score))))
print('Время обработки Логистической Регрессии {}'.format(t2-t1))
print('\n')

# Строим предсказание для тестовых данных:

for i in test.keys():
    test[i] = test[i].fillna(0)

t = test.as_matrix(columns=[x for x in train.keys() if x not in exclude and x not in categorial])
t = StandardScaler().fit_transform(t)

x = StandardScaler().fit_transform(x)

T_pick = np.zeros((test.shape[0], N))
for i, match_id in enumerate(test.index):
    for p in range(1, 6):
        X_pick[i, heroes[test.ix[match_id, 'r{}_hero'.format(p)]]] = 1
        X_pick[i, heroes[test.ix[match_id, 'd{}_hero'.format(p)]]] = -1

t = np.hstack((t, T_pick))

clf = LogisticRegression(**best[1])

# обучаем алгоритм на полных данных
clf.fit(x, y)

res = clf.predict_proba(t)[:, 1]

with open('result.txt', 'w') as f:
    f.write('match_id,radiant_win')
    for n, i in enumerate(res):
        s = '{},{}\n'.format(test.index[n], i)
        f.write(s)

print('Минимальное качество: {}'.format(min(res)))
print('Максимальное качество: {}'.format(max(res)))
