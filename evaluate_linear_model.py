import numpy as np
from glob import glob
import pandas as pd
from sklearn.metrics import roc_auc_score

base_dir = 'data/'
base_dir_linear = base_dir+'linear/'
base_dir_meta = base_dir+'meta/'

proba_list = []
for proba_file in glob(base_dir_meta+'*val*'):
    probas = []
    i = int(proba_file.split('.')[0][-1])+1
    for line in open(proba_file):
        if line.strip() != '':
            probas.append(float(line.strip()))
    proba_list.append([probas, i])

proba_list = np.concatenate([p[0] for p in sorted(proba_list, key=lambda x: x[1])])
gt_list = np.array(pd.read_csv(base_dir+'train_y.csv', header=None).ix[:, 0])

print(roc_auc_score(gt_list, proba_list, average='weighted'))


df_test = pd.read_csv(base_dir+'test1.csv')

proba_list = []
for proba_file in glob(base_dir_meta+'*test*'):
    probas = []
    for line in open(proba_file):
        if line.strip() != '':
            probas.append(float(line.strip()))
    proba_list.append(probas)

proba_list = np.array(proba_list)
proba_list = list(np.mean(proba_list, axis=0))

df_test['score'] = list(map(lambda x: float('%.6f' % x), proba_list))
df_test.to_csv(base_dir+'submission.csv', index=False)
