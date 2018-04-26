from scipy.sparse import load_npz
from collections import Counter

base_dir = 'data/'
X_train = load_npz(base_dir+'train_x.npz')

idx_non_zero = X_train.nonzero()
col_non_zero = idx_non_zero[1]
feature_count = Counter()
feature_count.update(col_non_zero)

feature_freq = []
for i in range(X_train.shape[1]):
    feature_freq.append((X_train.shape[0]-feature_count[i], feature_count[i]))

f_feature_freq = open(base_dir+'train_feature_freq.txt', 'w')
for freq in feature_freq:
    f_feature_freq.write(str(freq)+'\n')

