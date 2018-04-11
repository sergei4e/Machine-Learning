import pandas
from scipy.stats.stats import pearsonr
import numpy as np
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

print(data['Sex'].value_counts()['male'], data['Sex'].value_counts()['female'])

print(data['Survived'].value_counts() / data['Survived'].count() * 100)

print(data['Pclass'].value_counts() / data['Pclass'].count() * 100)

print(data['Age'].mean())

print(data['Age'].median())

print(pearsonr(data['SibSp'], data['Parch']))

#  (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).

sex = {'male': 1, 'female': 0}


def f(x):
    return [int(i) for i in x]


def g(x):
    return [float(i) for i in x]

new = pandas.DataFrame()
new['Pclass'] = data['Pclass']
new['Fare'] = data['Fare']
new['Age'] = data['Age']
new['Sex'] = [sex[i] for i in data['Sex']]
new['Survived'] = data['Survived']
new = new.dropna()


x = new.as_matrix(columns=['Pclass', 'Fare', 'Age', 'Sex'])
y = new.as_matrix(columns=['Survived'])


clf = DecisionTreeClassifier(random_state=241)
clf.fit(x, y)

print(clf.feature_importances_)
