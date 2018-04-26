from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from scipy.sparse import load_npz, csr_matrix
import pandas as pd
from multiprocessing import Pool, cpu_count
import math
import numpy as np
import time

base_dir = './data/'
base_dir_linear = base_dir+'linear/'

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

X_train = csr_matrix(load_npz(base_dir+'train_x.npz'))
y_train = np.array(pd.read_csv(base_dir + 'train_y.csv', header=None).ix[:, 0])
assert X_train.shape[0] == len(y_train)
split_idx_list = []
classifiers_with_id = []
for i in range(5):
    f_idx = open(base_dir+'fold_%d.idx' % i)
    split_idx = []
    for line in f_idx:
        if line.strip() != '':
            split_idx.append(int(line.strip()))
    split_idx_list.append(split_idx)
    clf = LogisticRegression(random_state=28, verbose=10, class_weight='balanced', solver='saga', max_iter=1000)
    classifiers_with_id.append([clf, i])


'''
for i in range(5):
    X = []
    y = []
    for j in range(5):
        if j != i:
            X.append(X_folds[j])
            y.append(np.array(y_folds[j]))
    X = vstack(X)
    y = np.concatenate(y)
    clf = LogisticRegression(random_state=28, verbose=10, class_weight='balanced', max_iter=1000, solver='saga')
    clf.fit(X, y)
    proba_eval = [sigmoid(x) for x in clf.decision_function(X_folds[i])]
    proba_test = [sigmoid(x) for x in clf.decision_function(X_test)]
    joblib.dump(clf, base_dir_linear + 'linear_model_part_%s.bin' % (i + 1))
    f_res_eval = open(base_dir_linear + 'linear_proba_val_%s.txt' % i, 'w')
    for proba in proba_eval:
        f_res_eval.write('%s\n' % proba)
    f_res_test = open(base_dir_linear + 'linear_proba_test_%s.txt' % i, 'w')
    for proba in proba_test:
        f_res_test.write('%s\n' % proba)
'''

def parallelize(classifiers_with_id, func):
    pool = Pool(cpu_count())
    res = pool.map(func, classifiers_with_id)
    pool.close()
    pool.join()
    return res

def fit_classifier(clf_id):
    clf, id = clf_id
    time.sleep(id*30)
    idx_list = []
    for i in range(5):
        if i != id:
            idx_list.append(np.array(split_idx_list[i]))
    idx_list = np.concatenate(idx_list)
    X = X_train[idx_list]
    y = y_train[idx_list]
    clf.fit(X, y)
    proba_eval = [sigmoid(x) for x in clf.decision_function(X_train[split_idx_list[id]])]
    #proba_test = [sigmoid(x) for x in clf.decision_function(X_test)]
    return clf, proba_eval, id

clf_proba_eval_id_list = sorted(parallelize(classifiers_with_id, fit_classifier), key=lambda x: x[2])
classifiers_fit_with_id = [[elem[0], elem[2]] for elem in clf_proba_eval_id_list]

X_test = csr_matrix(load_npz(base_dir+'test_x.npz'))

def predict_classifier(clf_id):
    clf, id = clf_id
    proba_test = [sigmoid(x) for x in clf.decision_function(X_test)]
    return proba_test, id

proba_test_list = sorted(parallelize(classifiers_fit_with_id, predict_classifier), key=lambda x: x[1])


for i in range(5):
    joblib.dump(clf_proba_eval_id_list[i][0], base_dir_linear+'linear_model_part_%s.bin' % (i+1))
    f_res_eval = open(base_dir_linear+'linear_proba_val_%s.txt' % i, 'w')
    for proba in clf_proba_eval_id_list[i][1]:
        f_res_eval.write('%s\n' % proba)
    f_res_test = open(base_dir_linear+'linear_proba_test_%s.txt' % i, 'w')
    for proba in proba_test_list[i][0]:
        f_res_test.write('%s\n' % proba)
