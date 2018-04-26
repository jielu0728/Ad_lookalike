from glob import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import math
from multiprocessing import Pool, cpu_count
from sklearn.externals import joblib

base_dir = 'data/'
base_dir_linear = base_dir+'linear/'
base_dir_meta = base_dir+'meta/'

def concat_proba_files(file_list):
    proba_with_id = []
    for f in file_list:
        idx = int(f.rsplit('.', 1)[0][-1])
        probas = []
        for line in open(f):
            if line.strip() != '':
                probas.append(float(line))
        proba_with_id.append([probas, idx])
    proba_with_id = np.concatenate([elem[0] for elem in sorted(proba_with_id, key=lambda x: x[1])])
    return proba_with_id

def mean_test_proba_files(file_list):
    X_test = []
    for f in file_list:
        probas = []
        for line in open(f):
            if line.strip() != '':
                probas.append(float(line))
        X_test.append(probas)
    return np.mean(X_test, axis=0)

proba_linear = concat_proba_files(glob(base_dir_linear+'*val*'))
proba_ffm = concat_proba_files(glob(base_dir+'*part*.res'))
proba_test_linear = mean_test_proba_files(glob(base_dir_linear+'*test*'))
proba_test_ffm = mean_test_proba_files(glob(base_dir+'*test*.res'))

X_train = np.column_stack([proba_linear, proba_ffm])
y_train = np.array(pd.read_csv(base_dir+'train_y.csv', header=None).ix[:, 0])
X_test = np.column_stack([proba_test_linear, proba_test_ffm])

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

print(X_train.shape)
print(y_train.shape)
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
    clf = LogisticRegression(random_state=28, verbose=10, class_weight='balanced', solver='saga', max_iter=500)
    classifiers_with_id.append([clf, i])


def parallelize(classifiers_with_id, func):
    pool = Pool(cpu_count())
    res = pool.map(func, classifiers_with_id)
    pool.close()
    pool.join()
    return res

def fit_classifier(clf_id):
    clf, id = clf_id
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

def predict_classifier(clf_id):
    clf, id = clf_id
    proba_test = [sigmoid(x) for x in clf.decision_function(X_test)]
    return proba_test, id

proba_test_list = sorted(parallelize(classifiers_fit_with_id, predict_classifier), key=lambda x: x[1])


for i in range(5):
    joblib.dump(clf_proba_eval_id_list[i][0], base_dir_meta+'stacking_model_part_%s.bin' % (i+1))
    f_res_eval = open(base_dir_meta+'stacking_proba_val_%s.txt' % i, 'w')
    for proba in clf_proba_eval_id_list[i][1]:
        f_res_eval.write('%s\n' % proba)
    f_res_test = open(base_dir_meta+'stacking_proba_test_%s.txt' % i, 'w')
    for proba in proba_test_list[i][0]:
        f_res_test.write('%s\n' % proba)
