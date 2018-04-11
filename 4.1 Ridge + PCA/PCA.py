from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr
import pandas
import numpy as np
import pdb

train = pandas.read_csv('close_prices.csv')
djia = pandas.read_csv('djia_index.csv')

columns = [x for x in train.keys() if x != 'date']

x = train.as_matrix(columns=columns)

pca = PCA(n_components=10)
X = pca.fit_transform(x)

count = 0
ss = 0

for i in pca.explained_variance_ratio_:
    if ss <= 0.9:
        ss += i
        count += 1

f1 = [x[0] for x in X]

koef = np.corrcoef(x=f1, y=djia['DJI'])
koef2 = pearsonr(x=f1, y=djia['DJI'])

ll = list(pca.components_[0])

nnn = ll.index(max(ll))

print(count)
print(koef)
print(koef2)
print(columns[nnn])


pdb.set_trace()
