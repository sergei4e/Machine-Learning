import pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, precision_recall_curve
import pdb

data = pandas.read_csv('classification.csv')

cls = pandas.read_csv('scores.csv')

TP, FP, FN, TN = 0, 0, 0, 0

for i in data.index:

    if data['pred'][i] == 1:
        if data['true'][i] == 1:
            TP += 1
        if data['true'][i] == 0:
            FP += 1
    if data['pred'][i] == 0:
        if data['true'][i] == 1:
            FN += 1
        if data['true'][i] == 0:
            TN += 1

print(TP, FP, FN, TN)

# TP, FP, FN, TN


A = accuracy_score(data['true'], data['pred'])
P = precision_score(data['true'], data['pred'])
R = recall_score(data['true'], data['pred'])
F = f1_score(data['true'], data['pred'])

print(A, P, R, F)

# true	score_logreg	score_svm	score_knn	score_tree


logreg = roc_auc_score(cls['true'], cls['score_logreg'])
svm = roc_auc_score(cls['true'], cls['score_svm'])
knn = roc_auc_score(cls['true'], cls['score_knn'])
tree = roc_auc_score(cls['true'], cls['score_tree'])

print(logreg, svm, knn, tree)

precision1, recall1, thresholds1 = precision_recall_curve(cls['true'], cls['score_logreg'])
precision2, recall2, thresholds2 = precision_recall_curve(cls['true'], cls['score_svm'])
precision3, recall3, thresholds3 = precision_recall_curve(cls['true'], cls['score_knn'])
precision4, recall4, thresholds4 = precision_recall_curve(cls['true'], cls['score_tree'])


def finder(precision, recall, thresholds):
    mas = []
    for n, r in enumerate(recall):
        if r >= 0.7:
            mas.append(precision[n])
    return max(mas)

print(finder(precision1, recall1, thresholds1))
print(finder(precision2, recall2, thresholds2))
print(finder(precision3, recall3, thresholds3))
print(finder(precision4, recall4, thresholds4))


pdb.set_trace()