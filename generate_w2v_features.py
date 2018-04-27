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
from gensim.models.word2vec import Word2Vec
from random import shuffle
import pickle

base_dir = './data/'
features_dir = base_dir+'feature/'
w2v_size = 10

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


w2v_feature = ['appIdAction', 'appIdInstall', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
data = data[w2v_feature+['label']]
w2v_feature = [[f, 'w2v'] for f in w2v_feature]
features = w2v_feature
shuffle(features)

train = data[data.label != -1]
train_y = train.pop('label')
#train, valid, train_y, valid_y = train_test_split(train, train_y, test_size=0.2, random_state=2018)
test = data[data.label == -1]
test = test.drop('label', axis=1)

def transform_w2v(w2v_model, rows):
    row_vectors = []
    for row in rows:
        row_vec = []
        for token in row.split(' '):
            if token in w2v_model.wv.vocab:
                row_vec.append(w2v_model.wv[token])
            else:
                row_vec.append(np.zeros(w2v_size))
        row_vec = np.mean(row_vec, axis=0)
        row_vectors.append(row_vec)
    return np.array(row_vectors)


def parallelize(features, func):
    pool = Pool(4)
    train_test_vectors_list = pool.map(func, features)
    pool.close()
    pool.join()
    feature_list = [v[1] for v in train_test_vectors_list]
    return [v[0][0] for v in train_test_vectors_list], [v[0][1] for v in train_test_vectors_list], feature_list

def transform_feature(feature):
    feature_name, func = feature
    if func == 'w2v':
        w2v = Word2Vec(data[feature_name].apply(lambda x: str(x).split(' ')), workers=1, size=w2v_size)
        train_test_vectors = (transform_w2v(w2v, train[feature_name]), transform_w2v(w2v, test[feature_name]))
        print('%s w2v prepared.' % feature_name)
        print('%s feature size: %d' % (feature_name, w2v_size))
        return train_test_vectors, '%s_%d' % (feature_name, w2v_size)

train_x, test_x, feature_list = parallelize(features, transform_feature)

for i in range(len(feature_list)):
    pickle.dump((train_x[i], test_x[i]), open(features_dir+'%s.ftr' % feature_list[i], 'wb'))
