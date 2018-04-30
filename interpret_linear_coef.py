from sklearn.externals import joblib
from glob import glob
import numpy as np

base_dir = 'coef/'

f_importance = open(base_dir+'feature_importance.txt')
importance = []
for line in f_importance:
    if line.strip() != '':
        importance.append(int(line.strip()))


feature_range = {}
f_feature = open(base_dir+'feature_selection_order.txt')
start = 0
count_ = 0
for line in f_feature:
    if line.strip() != '':
        count = int(line.strip().rsplit('_')[-2])
        end = start + count
        for i in range(start, end):
            feature_range[i] = line.strip().rsplit('_', 2)[0]
        start = end
        count_ += count

assert len(importance) == count_

f = open(base_dir+'interpret_coef.txt', 'w')
for i in range(len(importance)):
    f.write('%s_%d: %s\n' % (feature_range[i], i, importance[i]))

feature_importance_avg = {}
feature_importance_sum = {}
for i in range(len(importance)):
    if feature_range[i] not in feature_importance_avg:
        feature_importance_avg[feature_range[i]] = [importance[i]]
        feature_importance_sum[feature_range[i]] = [importance[i]]
    else:
        feature_importance_avg[feature_range[i]].append(importance[i])
        feature_importance_sum[feature_range[i]].append(importance[i])


for k in feature_importance_avg.keys():
    feature_importance_avg[k] = np.mean(feature_importance_avg[k])
    feature_importance_sum[k] = sum(feature_importance_sum[k])

feature_importance_sorted = sorted(feature_importance_avg.items(), key=lambda x: x[1], reverse=True)
feature_importance_sum_sorted = sorted(feature_importance_sum.items(), key=lambda x: x[1], reverse=True)
low_importance = set()
for k, v in feature_importance_sorted:
    if v < 0.1:
        low_importance.add(k)
    print(k, v)

for k, v in feature_importance_sum_sorted:
    print(k, v)

joblib.dump(low_importance, 'coef/low_importance.set')
