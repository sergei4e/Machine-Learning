import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
import pdb

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space'],
                    download_if_missing=False)

vec = TfidfVectorizer()
data = vec.fit_transform(newsgroups.data)

words = vec.get_feature_names()

x = data.toarray()  # [:200]
y = newsgroups.target  # [:200]

# grid = {'C': np.power(10.0, np.arange(-5, 6))}
# clf = SVC(kernel='linear', random_state=241)

# kf = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
# gs = GridSearchCV(clf, grid, scoring='accuracy', cv=kf)
# gs.fit(x, y)

clf1 = SVC(C=1.0, kernel='linear', random_state=241)
clf1.fit(x, y)

params = clf1.coef_[0]

print(len(words))
print(len(params))

vse = [(w, params[n]) for n, w in enumerate(words)]
vse = sorted(vse, key=(lambda x: abs(x[1])), reverse=True)[:10]

w = [x[0] for x in vse]

w.sort()

print(w)

# pdb.set_trace()
