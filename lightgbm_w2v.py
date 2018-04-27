import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import hstack, csr_matrix
import os
from multiprocessing import cpu_count, Pool
from sklearn.externals import joblib
import numpy as np
from glob import glob
from random import shuffle
import pickle
import sys

base_dir = './data/'
feature_dir = base_dir+'feature/'
w2v_model_dir = base_dir+'lgb_w2v/'
w2v_size = 20
i_th_fold = int(sys.argv[1])
'''
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

one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house',
                   'advertiserId', 'campaignId', 'creativeId', 'adCategoryId',
                   'productId', 'productType', 'creativeSize']

cv_feature = ['marriageStatus', 'ct', 'os', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5']

#w2v_feature = ['appIdAction', 'appIdInstall', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']


one_hot_feature = [[f, 'one-hot'] for f in one_hot_feature]
cv_feature = [[f, 'cv'] for f in cv_feature]
features = one_hot_feature + cv_feature
shuffle(features)

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
    func_list = [v[2] for v in train_test_vectors_list]
    return [v[0][0] for v in train_test_vectors_list], [v[0][1] for v in train_test_vectors_list], feature_list, func_list

def transform_feature(feature):
    feature_name, func = feature
    if func == 'cv':
        cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
        cv.fit(data[feature_name])
        train_test_vectors = (cv.transform(train[feature_name]), cv.transform(test[feature_name]))
        print('%s cv prepared.' % feature_name)
        print('%s feature size: %d' % (feature_name, len(cv.vocabulary_)))
        return train_test_vectors, '%s_%d' % (feature_name, len(cv.vocabulary_)), func
    elif func == "one-hot":
        enc = LabelBinarizer(sparse_output=True)
        enc.fit(data[feature_name])
        train_test_encodes = (enc.transform(train[feature_name]), enc.transform(test[feature_name]))
        print('%s one-hot prepared.' % feature_name)
        print('%s feature size: %d' % (feature_name, len(enc.classes_)))
        return train_test_encodes, '%s_%d' % (feature_name, len(enc.classes_)), func

train_x_list, test_x_list, feature_list, func_list = parallelize(features, transform_feature)

for i in range(len(train_x_list)):
    pickle.dump((train_x_list[i], test_x_list[i]),
                open(feature_dir+'%s_%s.ftr' % (feature_list[i], func_list[i]), 'wb'))
'''

train_y = np.array(pd.read_csv(base_dir + 'train_y.csv', header=None).ix[:, 0])
res = pd.read_csv(base_dir+'test1.csv')

feature_list = []
train_x = []
test_x = []
for f_train_w2v_part in glob(feature_dir+'*.ftr'):
    feature_name = f_train_w2v_part.split('/')[-1].split('.')[0].rsplit('_', 1)[0]
    train_x_part, test_x_part = pickle.load(open(f_train_w2v_part, 'rb'))
    train_x.append(train_x_part)
    test_x.append(test_x_part)
    feature_list.append(feature_name)

print(feature_list)
train_x = csr_matrix(hstack(train_x))
test_x = csr_matrix(hstack(test_x))
print(train_x.shape)

split_idx_list = []
for i in range(5):
    f_idx = open(base_dir+'fold_%d.idx' % i)
    split_idx = []
    for line in f_idx:
        if line.strip() != '':
            split_idx.append(int(line.strip()))
    split_idx_list.append(split_idx)

valid_x, valid_y = train_x[split_idx_list[i_th_fold-1]], train_y[split_idx_list[i_th_fold-1]]
train_idx = []
for i in range(5):
    if i != i_th_fold-1:
        train_idx += split_idx_list[i]
train_x, train_y = train_x[train_idx], train_y[train_idx]

f_feature_list = open(base_dir + 'feature_list_w2v.txt', 'w')
for f in feature_list:
    f_feature_list.write('%s\n' % f)
f_feature_list.close()

def LGB_predict(train_x, train_y, valid_x, valid_y, test_x, res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=10000, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric='auc', early_stopping_rounds=1000)
    joblib.dump(clf, w2v_model_dir + 'lgb_w2v_part_%s.model' % i_th_fold)
    res['score'] = clf.predict_proba(test_x)[:, 1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv(w2v_model_dir+'submission_part_%s.csv' % i_th_fold, index=False)
    f_proba_eval = open(w2v_model_dir+'lgb_proba_val_%s.txt' % i_th_fold, 'w')
    for p in clf.predict_proba(valid_x)[:, 1]:
        f_proba_eval.write('%.6f\n' % p)
    f = open(w2v_model_dir+'feature_importance_w2v_part_%s.txt' % i_th_fold, 'w')
    for fi in clf.feature_importances_:
        f.write('%s\n' % fi)
    f.close()
    return clf

model = LGB_predict(train_x, train_y, valid_x, valid_y, test_x, res)
