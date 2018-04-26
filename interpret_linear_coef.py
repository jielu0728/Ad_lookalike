from sklearn.externals import joblib
from glob import glob
import numpy as np

base_dir = 'coef/'

coef_list = []
for f_model in glob(base_dir+'*model*'):
    coef_list.append(joblib.load(f_model).coef_)
coef = np.mean(coef_list, axis=0)[0]

feature_range = {}
f_feature = open(base_dir+'feature_list.txt')
start = 0
for line in f_feature:
    count = int(line.strip().split('_')[1])
    end = start + count
    for i in range(start, end):
        feature_range[i] = line.strip().split('_')[0]
    start = end

f = open(base_dir+'interpret_coef.txt', 'w')
feature_name = ''
for i in range(len(coef)):
    if feature_range[i] != feature_name:
        f.write('-------------------------------------\n')
    feature_name = feature_range[i]
    f.write('%s_%d: %s\n' % (feature_range[i], i, coef[i]))