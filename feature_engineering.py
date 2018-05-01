from tqdm import tqdm
from collections import Counter
import numpy as np
def combine_features(feature_names_left, feature_names_right, data, take_limit):
    new_cols = {}
    dict_limited = {}
    data_limited = {}
    feature_size = 0
    if take_limit is not None:
        bar0 = tqdm(total=len(feature_names_left+feature_names_right)+1, position=0)
        for f in feature_names_left+feature_names_right:
            dict_limited[f] = set(data[f].value_counts().nlargest(take_limit).index)
            data_limited[f] = data[f].apply(lambda v: str(v) if v in dict_limited[f] else 'n')
            bar0.update()
    else:
        bar0 = tqdm(total=len(feature_names_left+feature_names_right) + 1, position=0)
        for f in feature_names_left+feature_names_right:
            data_limited[f] = data[f].apply(str)
            bar0.update()
    bar = tqdm(total=len(feature_names_left)*len(feature_names_right), position=1)
    for i in range(len(feature_names_left)):
        for j in range(len(feature_names_right)):
            combined_name = '%s_%s' % (feature_names_left[i], feature_names_right[j])
            new_cols[combined_name] = (data_limited[feature_names_left[i]] + '_' + data_limited[feature_names_right[j]]).astype('category').cat.codes
            feature_size += len(data_limited[feature_names_left[i]].value_counts()) * len(data_limited[feature_names_right[j]].value_counts())
            bar.update()
    print('Estimated combined feature size: %d' % feature_size)
    return new_cols

class PercentageTransformer:
    def __init__(self):
        self.counter = Counter()

    def fit(self, list):
        self.counter.update(list)
        self.total = sum(self.counter.values())
        self.counter = {k: v/self.total for k, v in self.counter.items()}

    def transform(self, list):
        return [self.counter[elem] if elem in self.counter else 0. for elem in list]

    def fit_transform(self, list):
        self.fit(list)
        return self.transform(list)

    def fit_seq(self, list_of_seq):
        for line in list_of_seq:
            self.counter.update(line.split(' '))
        self.total = sum(self.counter.values())
        self.counter = {k: v/self.total for k, v in self.counter.items()}

    def transform_seq(self, list_of_seq):
        transformed_seq = []
        for line in list_of_seq:
            transformed_seq.append(np.mean([self.counter[token] if token in self.counter else 0 for token in line.split(' ')]))
        return transformed_seq

    def fit_transform_seq(self, list_of_seq):
        self.fit_seq(list_of_seq)
        return self.transform_seq(list_of_seq)