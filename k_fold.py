import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import hstack, save_npz, load_npz, csr_matrix
import os
from multiprocessing import cpu_count, Pool
from sklearn.externals import joblib
import numpy as np

base_dir = './data/'

'''
if os.path.exists(base_dir + 'data.csv'):
    data = pd.read_csv(base_dir + 'data.csv')
else:
    ad_feature = pd.read_csv(base_dir + 'adFeature.csv')
    if os.path.exists(base_dir + 'userFeature.csv'):
        user_feature = pd.read_csv(base_dir + 'userFeature.csv')
    else:
        userFeature_data = []
        with open(base_dir + 'userFeature.data') as f:
            bar = tqdm(total=11420040)
            for i, line in enumerate(f):
                line = line.strip().split('|')
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ', 1)
                    userFeature_dict[each_list[0]] = each_list[1]
                userFeature_data.append(userFeature_dict)
                bar.update()
        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv(base_dir + 'userFeature.csv', index=False)
        print('User Feature saved')

    train = pd.read_csv(base_dir + 'train.csv')
    predict = pd.read_csv(base_dir + 'test1.csv')
    train.loc[train['label'] == -1, 'label'] = 0
    predict['label'] = -1
    data = pd.concat([train, predict])
    data = pd.merge(data, ad_feature, on='aid', how='left')
    data = pd.merge(data, user_feature, on='uid', how='left')
    data.fillna('-1', inplace=True)
    data.to_csv(base_dir + 'data.csv', index=False)

one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os',
                   'ct', 'marriageStatus', 'advertiserId', 'campaignId', 'creativeId', 'adCategoryId',
                   'productId', 'productType', 'creativeSize']

cv_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4',
              'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

one_hot_feature = [[f, 'one-hot'] for f in one_hot_feature]
cv_feature = [[f, 'cv'] for f in cv_feature]
features = one_hot_feature + cv_feature

train = data[data.label != -1]
train_y = train.pop('label')
test = data[data.label == -1]
res = test[['aid', 'uid']]
test = test.drop('label', axis=1)


def parallelize(features, func):
    pool = Pool(cpu_count())
    train_test_vectors_list = pool.map(func, features)
    pool.close()
    pool.join()
    feature_list = [v[1] for v in train_test_vectors_list]
    return hstack([v[0][0] for v in train_test_vectors_list]), hstack([v[0][1] for v in train_test_vectors_list]), feature_list


def transform_feature(feature):
    feature_name, func = feature
    if func == 'cv':
        cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
        cv.fit(data[feature_name])
        train_test_vectors = (cv.transform(train[feature_name]), cv.transform(test[feature_name]))
        print('%s cv prepared.' % feature_name)
        print('%s feature size: %d' % (feature_name, len(cv.vocabulary_)))
        return train_test_vectors, '%s_%d' % (feature_name, len(cv.vocabulary_))
    elif func == "one-hot":
        enc = LabelBinarizer(sparse_output=True)
        enc.fit(data[feature_name])
        train_test_encodes = (enc.transform(train[feature_name]), enc.transform(test[feature_name]))
        print('%s one-hot prepared.' % feature_name)
        print('%s feature size: %d' % (feature_name, len(enc.classes_)))
        return train_test_encodes, '%s_%d' % (feature_name, len(enc.classes_))

train_x, test_x, feature_list = parallelize(features, transform_feature)
save_npz(base_dir + 'train_x.npz', train_x)
save_npz(base_dir + 'test_x.npz', test_x)
train_y.to_csv(base_dir + 'train_y.csv', index=False)
f_feature_list = open(base_dir + 'feature_list.txt', 'w')
for f in feature_list:
    f_feature_list.write('%s\n' % f)
f_feature_list.close()
'''

train_x = csr_matrix(load_npz(base_dir + 'train_x.npz'))
train_y = pd.read_csv(base_dir + 'train_y.csv', header=None)
test_x = load_npz(base_dir + 'test_x.npz')

i = 0
idx_set = set()
kf = KFold(n_splits=5, shuffle=True, random_state=2018)
for _, part_idx in kf.split(train_x):
    f_idx = open(base_dir+'fold_%d.idx' % i, 'w')
    for idx in part_idx:
        f_idx.write('%d\n' % idx)
    train_x_split = train_x[part_idx]
    train_y_split = train_y.iloc[part_idx]
    save_npz(base_dir + 'train_x_part_%d.npz' % (i + 1), train_x_split)
    train_y_split.to_csv(base_dir + 'train_y_part_%d.csv' % (i + 1), index=False)
    idx_set.update(part_idx)
    print('Part %d saved, len: %d' % ((i + 1), len(train_y_split)))
    i += 1

if len(idx_set) == len(train_y):
    print('No overlap')
else:
    print('Overlap catched')
