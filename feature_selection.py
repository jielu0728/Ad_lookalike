import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import hstack, csr_matrix, lil_matrix, find, dok_matrix, save_npz, csc_matrix, bsr_matrix
import os
from multiprocessing import cpu_count, Pool
from sklearn.externals import joblib
import numpy as np
from glob import glob
from random import shuffle, sample
import pickle
import sys
from feature_engineering import combine_features, PercentageTransformer

base_dir = './data/'
feature_selection_dir = base_dir+'feature_selection/'
'''
split_idx_list = []
for i in range(5):
    f_idx = open(base_dir+'fold_%d.idx' % i)
    split_idx = []
    for line in f_idx:
        if line.strip() != '':
            split_idx.append(int(line.strip()))
    split_idx_list.append(split_idx)

train_idx = []
for i in range(1, 5):
    train_idx += sample(split_idx_list[i], int(0.05*len(split_idx_list[i])))
    #train_idx += split_idx_list[i]
valid_idx = sample(split_idx_list[0], int(0.05*len(split_idx_list[0])))
#valid_idx = split_idx_list[0]
assert len(set(train_idx) & set(valid_idx)) == 0

if os.path.exists(base_dir+'data.csv'):
    data = pd.read_csv(base_dir+'data.csv')
else:
    ad_feature = pd.read_csv(base_dir+'adFeature.csv')
    if os.path.exists(base_dir+'userFeature.csv'):
        user_feature = pd.read_csv(base_dir+'userFeature.csv')
    else:
        userFeature_data = []
        with open(base_dir+'userFeature.data') as f:
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
        user_feature.to_csv(base_dir+'userFeature.csv', index=False)
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
print('Data loaded')

train = data[data.label != -1]
train_y = train.pop('label')
train = train.ix[train_idx+valid_idx]
train_y = train_y.ix[train_idx+valid_idx]

joblib.dump((train_idx, valid_idx), feature_selection_dir+'train_valid_idx.bin')

one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house',
                   'advertiserId', 'campaignId', 'creativeId', 'adCategoryId',
                   'productId', 'productType', 'creativeSize', 'marriageStatus', 'ct', 'os']

cv_feature = ['interest1', 'interest2', 'interest3', 'interest4',
              'interest5', 'appIdAction', 'appIdInstall', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

one_hot_combined_feature = []

for feature_name, col in combine_features(one_hot_feature, train, None).items():
    one_hot_combined_feature.append(feature_name)
    train[feature_name] = col

print(train)
one_hot_feature = [[f, 'one-hot'] for f in one_hot_feature]
cv_feature = [[f, 'cv'] for f in cv_feature]
one_hot_combined_feature = [[f, 'one-hot'] for f in one_hot_combined_feature]
features = one_hot_feature + cv_feature + one_hot_combined_feature
shuffle(features)

train_x = train.ix[train_idx]
valid_x = train.ix[valid_idx]
train_y = train_y.ix[train_idx]
valid_y = train_y.ix[valid_idx]

def parallelize(features, func):
    pool = Pool(cpu_count())
    pool.map(func, features)
    pool.close()
    pool.join()

def transform_feature(feature):
    feature_name, func = feature
    if func == 'cv':
        cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
        pt = PercentageTransformer()
        cv.fit(train[feature_name])
        pt.fit_seq(train[feature_name])
        train_vectors = cv.transform(train_x[feature_name])
        train_stats_vectors = pt.transform_seq(train_x[feature_name])
        valid_vectors = cv.transform(valid_x[feature_name])
        valid_stats_vectors = pt.transform_seq(valid_x[feature_name])
        print('%s cv prepared.' % feature_name)
        print('%s feature size: %d' % (feature_name, train_vectors.shape[1]))
        pickle.dump((train_vectors, valid_vectors),
                    open(feature_selection_dir + '%s_%s.ftr' % ('%s_%d' % (feature_name, train_vectors.shape[1]), func), 'wb'))
        pickle.dump((train_stats_vectors, valid_stats_vectors),
                    open(feature_selection_dir + '%s_%s.ftr' % ('%s_%d' % (feature_name+'Stats', 1), func),
                         'wb'))
    elif func == "one-hot":
        enc = LabelBinarizer(sparse_output=True)
        pt = PercentageTransformer()
        enc.fit(train[feature_name])
        pt.fit(train[feature_name])
        train_encodes = enc.transform(train_x[feature_name])
        train_stats_encodes = pt.transform(train_x[feature_name])
        valid_encodes = enc.transform(valid_x[feature_name])
        valid_stats_encodes = pt.transform(valid_x[feature_name])
        print('%s one-hot prepared.' % feature_name)
        print('%s feature size: %d' % (feature_name, train_encodes.shape[1]))
        pickle.dump((train_encodes, valid_encodes),
                    open(feature_selection_dir+'%s_%s.ftr' % ('%s_%d' % (feature_name, train_encodes.shape[1]), func), 'wb'))
        pickle.dump((train_stats_encodes, valid_stats_encodes),
                    open(feature_selection_dir + '%s_%s.ftr' % ('%s_%d' % (feature_name+'Stats', 1), func),
                         'wb'))

parallelize(features, transform_feature)
'''
train_idx, valid_idx = joblib.load(feature_selection_dir+'train_valid_idx.bin')

y = pd.read_csv(base_dir + 'train_y.csv', header=None).ix[:, 0]
train_y, valid_y = y[train_idx], y[valid_idx]

low_importance_set = joblib.load(feature_selection_dir+'low_importance.set')

feature_list = []
f_feature = glob(feature_selection_dir+'*.ftr')
f_feature = sorted(f_feature, key=lambda x: int(x.split('/')[-1].split('_')[-2]))
bar = tqdm(total=len(f_feature)+1)
train_x_sparse = []
valid_x_sparse = []
train_x_list = []
valid_x_list = []
feature_list_sparse = []
feature_list_list = []
feature_count = 0
for f_train_part in f_feature:
    feature_name = f_train_part.split('/')[-1].split('.')[0]
    tmp = f_train_part.split('/')[-1].rsplit('_', 2)[0]
    if tmp in low_importance_set:
        print('Feature: %s is low importance' % feature_name)
    else:
        train_x_part, valid_x_part = pickle.load(open(f_train_part, 'rb'))
        if type(train_x_part) == list:
            train_x_list.append(train_x_part)
            valid_x_list.append(valid_x_part)
            feature_list_list.append(feature_name)
            feature_count += 1
        else:
            train_x_sparse.append(train_x_part)
            valid_x_sparse.append(valid_x_part)
            feature_list_sparse.append(feature_name)
            feature_count += train_x_part.shape[1]
    bar.update()

if train_x_list != []:
    train_x_list = np.column_stack(train_x_list)
    valid_x_list = np.column_stack(valid_x_list)
    print(train_x_list.shape)
    print(valid_x_list.shape)

train_x_sparse = hstack(train_x_sparse)
valid_x_sparse = hstack(valid_x_sparse)
print(train_x_sparse.shape)
print(valid_x_sparse.shape)

if train_x_list != []:
    train_x = csc_matrix(hstack([train_x_sparse, train_x_list]))
    valid_x = csc_matrix(hstack([valid_x_sparse, valid_x_list]))
else:
    train_x = csc_matrix(train_x_sparse)
    valid_x = csc_matrix(valid_x_sparse)
feature_list = feature_list_sparse + feature_list_list
print(feature_list)
assert feature_count == train_x.shape[1]

f_feature_list = open(feature_selection_dir + 'feature_selection_order.txt', 'w')
for f in feature_list:
    f_feature_list.write('%s\n' % f)
f_feature_list.close()

def LGB_predict(train_x, train_y, valid_x, valid_y):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=10000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric='auc', early_stopping_rounds=500)
    joblib.dump(clf, feature_selection_dir + 'lgb_feature_selection.model')
    f = open(feature_selection_dir+'feature_importance.txt', 'w')
    for fi in clf.feature_importances_:
        f.write('%s\n' % fi)
    f.close()
    return clf

model = LGB_predict(train_x, train_y, valid_x, valid_y)
print(train_x.shape)
print(valid_x.shape)
