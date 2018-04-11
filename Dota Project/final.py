import pandas
import numpy as np
import math
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from pprint import pprint
from itertools import izip
from scipy.sparse import coo_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import pandas
from sklearn.utils import shuffle
from sklearn import grid_search
import time
import datetime
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression 

future_features = ['duration','radiant_win','tower_status_radiant','tower_status_dire','barracks_status_radiant','barracks_status_dire']

categorical_features = ['lobby_type','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero']

def filter_features(heads, exclude):
    return list(filter(lambda x: x not in exclude, heads))

def print_features_with_gaps(heads, features):
    for h in heads:
        if features[h].count() != len(features.index):
            print h

def try_grad_boost_with_30_estimators(xtrain, ytrain):
    start_time = datetime.datetime.now()
    gbt = GradientBoostingClassifier(n_estimators=30, learning_rate=0.4)
    shuffle = cross_validation.KFold(len(xtrain), n_folds=5, shuffle=True, random_state=241)
    scores = cross_validation.cross_val_score(gbt, xtrain, ytrain, scoring="roc_auc", cv=shuffle)
    return scores.mean(), datetime.datetime.now() - start_time

def get_best_estimators_for_grad_boost(xtrain, ytrain):
    gbt = GradientBoostingClassifier()
    param_grid = {"n_estimators": [10,20,30,40], 'learning_rate': [0.1, 0.2, 0.3, 0.4,0.5]}

    clf = grid_search.GridSearchCV(gbt, param_grid, cv=5, scoring='roc_auc')
    _ = clf.fit(xtrain, ytrain)
    return clf.best_estimator_

def try_log_reg_with_best_param_C(xtrain, ytrain):
    start_time = datetime.datetime.now()
    lreg = LogisticRegression(penalty='l2', C=0.01)

    shuffle = cross_validation.KFold(len(xtrain), n_folds=5, shuffle=True, random_state=241)
    scores = cross_validation.cross_val_score(lreg, xtrain, ytrain, scoring="roc_auc", cv=shuffle)
    return scores.mean(), datetime.datetime.now() - start_time

def get_best_estimators_for_log_reg(xtrain, ytrain):
    lreg = LogisticRegression(penalty='l2')
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

    clf = grid_search.GridSearchCV(lreg, param_grid, cv=5, scoring='roc_auc')
    _ = clf.fit(xtrain, ytrain)
    return clf.best_estimator_

def calculate_distinct_heroes_ids_count():
    ft = pandas.read_csv('features.csv', index_col='match_id')
    frames = [ft['r1_hero'],ft['r2_hero'],ft['r3_hero'],ft['r4_hero'],ft['r5_hero'],ft['d1_hero'],ft['d2_hero'],ft['d3_hero'],ft['d4_hero'],ft['d5_hero']]
    result = pandas.concat(frames)
    return len(result.value_counts())

# GradientBoosting
features = pandas.read_csv('features.csv', index_col='match_id')
heads_for_train = filter_features(list(features.columns.values), future_features)
print_features_with_gaps(heads_for_train, features)

# 1.
#first_blood_time first_blood_team first_blood_player1 first_blood_player2 
#radiant_bottle_time radiant_courier_time radiant_flying_courier_time radiant_first_ward_time 
#dire_bottle_time dire_courier_time dire_flying_courier_time dire_first_ward_time
#
#for example first_blood_time is not mandatory event in Dota, consequently it can be nan.
#similarly radiant_bottle_time==nan means that bottle wasn't bought  

# 2. 
#    radiant_win contains variable to predict.

features = features.fillna(0)

xtrain = np.array(features[heads_for_train])
ytrain = np.array(features['radiant_win'])

auc, timediff = try_grad_boost_with_30_estimators(xtrain, ytrain)
print 'Auc roc : ', auc, 'Time elapsed : ', timediff
# 3. 
#    Auc roc :  0.702168571994 Time elapsed :  0:04:29.363304

print get_best_estimators_for_grad_boost(xtrain, ytrain)
# 4.
#    prints learning_rate=0.4 n_estimators=40
#    it means that GradientBoosting with 40 estimators works better than with 30
#    usualy optimal amount of estimators lies between 100 and 200
#    to speed up GradientBoosting you can set max_depth and subsample params less than default


# LogisticRegression
features = pandas.read_csv('features.csv', index_col='match_id')
heads_for_train = filter_features(list(features.columns.values), future_features)
features = features.fillna(0)
xtrain = np.array(features[heads_for_train])
ytrain = np.array(features['radiant_win'])
xtrain = StandardScaler().fit_transform(xtrain)

print get_best_estimators_for_log_reg(xtrain, ytrain)
# optimal hyperparam is C=0.01

auc, timediff = try_log_reg_with_best_param_C(xtrain, ytrain)
print 'Auc roc : ', auc, 'Time elapsed : ', timediff
# 1.
#    Auc roc :  0.716341465365 Time elapsed :  0:00:15.420653
#    LogisticRegression has auc roc a bit more then auc roc of gradient boosting
#    It caused by bad preparation of features, bad tuning of gradient boosting,
#    it have to be learned with more estimators and tuning of other hyper params
#    We fill nan with 0 but it is good for regresssion and not for boosting,
#    it is better to fill nan with value like -999 for gradient boosting
#    LogisticRegression works 20 times faster than gradient boosting

heads_for_train = filter_features(list(features.columns.values), future_features + categorical_features)
xtrain = np.array(features[heads_for_train])
xtrain = StandardScaler().fit_transform(xtrain)

print get_best_estimators_for_log_reg(xtrain, ytrain)
# optimal hyperparam is C=0.01

auc, timediff = try_log_reg_with_best_param_C(xtrain, ytrain)
print 'Auc roc : ', auc, 'Time elapsed : ', timediff

# 2.
#    Auc roc :  0.716400950653 Time elapsed :  0:00:14.103887
#    auc roc slightly inreased because we deleted bad noise features

print calculate_distinct_heroes_ids_count()
# 3.
#    108 unique heroes.

def prepare_features_with_heroes_dictionary(features):
    heads_for_train = filter_features(list(features.columns.values), future_features + categorical_features)
    features = features.fillna(0)
    xtrain = np.array(features[heads_for_train])


    X_pick = np.zeros((features.shape[0], 112))

    for i, match_id in enumerate(features.index):
        for p in xrange(5):
            X_pick[i, features.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, features.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

    x_full = np.hstack((xtrain,X_pick))
    return x_full

scaler = StandardScaler()
features = pandas.read_csv('features.csv', index_col='match_id')
train_x = prepare_features_with_heroes_dictionary(features)
train_y = np.array(features['radiant_win'])
train_x = scaler.fit_transform(train_x)

print get_best_estimators_for_log_reg(train_x, train_y)
# optimal hyperparam is C=0.01
auc, timediff = try_log_reg_with_best_param_C(train_x, train_y)
print 'Auc roc : ', auc, 'Time elapsed : ', timediff


# 4.
#    Auc roc :  0.751970490091 Time elapsed :  0:00:26.698953
#    Auc roc increased greatly, because heroes features are strong - 
#    one hero could be stronger than another

features = pandas.read_csv('features_test.csv', index_col='match_id')
test_x = prepare_features_with_heroes_dictionary(features)
test_x = scaler.transform(test_x)

lreg = LogisticRegression(penalty='l2', C=0.01)
lreg.fit(train_x, train_y)

df = pandas.DataFrame(index=features.index, columns=['target'])
predictions = lreg.predict_proba(test_x)[:, 1]
df['target'] = predictions
df.to_csv('predictions.csv')

print 'Min prediction : ', min(predictions), ' Max prediction : ', max(predictions)

# 5.
#    Min prediction :  0.00849095194724 Max prediction :  0.996277624036
