import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import hstack
import os
from multiprocessing import cpu_count, Pool
from sklearn.externals import joblib
import numpy as np

base_dir = './data/'
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
train, valid, train_y, valid_y = train_test_split(train, train_y, test_size=0.2, random_state=2018)
test = data[data.label == -1]
res = test[['aid', 'uid']]
test = test.drop('label', axis=1)


def parallelize(features, func):
    pool = Pool(cpu_count())
    train_valid_test_vectors_list = pool.map(func, features)
    pool.close()
    pool.join()
    return hstack([v[0] for v in train_valid_test_vectors_list]), hstack([v[1] for v in train_valid_test_vectors_list]), \
           hstack([v[2] for v in train_valid_test_vectors_list])

def transform_feature(feature):
    feature_name, func = feature
    if func == 'cv':
        cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
        cv.fit(data[feature_name])
        train_test_vectors = (cv.transform(train[feature_name]), cv.transform(valid[feature_name]),
                              cv.transform(test[feature_name]))
        print('%s cv prepared.' % feature_name)
        print('%s feature size: %d' % (feature_name, len(cv.vocabulary_)))
        return train_test_vectors
    elif func == "one-hot":
        enc = LabelBinarizer(sparse_output=True)
        enc.fit(data[feature_name])
        train_test_encodes = (enc.transform(train[feature_name]), enc.transform(valid[feature_name]),
                              enc.transform(test[feature_name]))
        print('%s one-hot prepared.' % feature_name)
        print('%s feature size: %d' % (feature_name, len(enc.classes_)))
        return train_test_encodes

train_x, valid_x, test_x = parallelize(features, transform_feature)


def LGB_test(train_x, train_y, test_x, test_y):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=28, n_jobs=cpu_count()
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], eval_metric='auc', early_stopping_rounds=100)
    # print(clf.feature_importances_)
    return clf, clf.best_score_['valid_1']['auc']


def LGB_predict(train_x, train_y, valid_x, valid_y, test_x, res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=10000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric='auc', early_stopping_rounds=1000)
    joblib.dump(clf, base_dir + 'lgb.model')
    res['score'] = clf.predict_proba(test_x)[:, 1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv(base_dir+'submission.csv', index=False)
    print(list(clf.feature_importances_))
    return clf

model = LGB_predict(train_x, train_y, valid_x, valid_y, test_x, res)

