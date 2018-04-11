import pandas
import pdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

train = pandas.read_csv('salary-train.csv')
test = pandas.read_csv('salary-test-mini.csv')

# 'FullDescription', 'LocationNormalized', 'ContractTime', 'SalaryNormalized'

train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
test['LocationNormalized'].fillna('nan', inplace=True)
test['ContractTime'].fillna('nan', inplace=True)


enc = DictVectorizer()
X_train = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))

vec = TfidfVectorizer(min_df=5)
data = vec.fit_transform(train['FullDescription'])
data2 = vec.transform(test['FullDescription'])

words = vec.get_feature_names()

# x_w = data.toarray()  # [:200]

x = hstack([data, X_train])
X = hstack([data2, X_test])
y = train['SalaryNormalized']  # [:200]

pdb.set_trace()

clf = Ridge(random_state=241, alpha=1)
clf.fit(x, y)
res = clf.predict(X)

print(res)

pdb.set_trace()

x = data.as_matrix(columns=['f1', 'f2'])
y = data['obj']