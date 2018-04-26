from scipy.sparse import load_npz, find
import pandas as pd
import sys
from tqdm import tqdm

base_dir = './data/'

feature_range = {}
f_feature = open(base_dir+'feature_list.txt')
start = 0
feature_id = 0
for line in f_feature:
    count = int(line.strip().split('_')[1])
    end = start + count
    for i in range(start, end):
        feature_range[i] = feature_id
    start = end
    feature_id += 1

train_or_test = sys.argv[1]


if train_or_test == 'train':
    sparse_matrix = load_npz(sys.argv[2])
    labels = list(pd.read_csv(sys.argv[3]).ix[:, 0])
    print('len: %d' % len(labels))
    print('generating: %s' % base_dir+'libffm_part_%s.txt' % sys.argv[1].split('.')[0][-1])
    f_libffm = open(base_dir+'libffm_part_%s.txt' % sys.argv[1].split('.')[0][-1], 'w')
elif train_or_test == 'test':
    sparse_matrix = load_npz(sys.argv[2])
    print('len: %d' % sparse_matrix.shape[0])
    print('generating: %s' % base_dir+'libffm_test.txt')
    f_libffm = open(base_dir+'libffm_test.txt', 'w')
else:
    raise Exception('train_or_test arg error')

x_pos, y_pos, value = find(sparse_matrix)
non_zero_values_by_line = {}

bar = tqdm(total=len(x_pos))
for i in range(len(x_pos)):
    if x_pos[i] not in non_zero_values_by_line:
        non_zero_values_by_line[x_pos[i]] = [(y_pos[i], value[i])]
    else:
        non_zero_values_by_line[x_pos[i]].append((y_pos[i], value[i]))
    bar.update()

for i in range(sparse_matrix.shape[0]):
    non_zero_values_line_i = sorted(non_zero_values_by_line[i], key=lambda x: x[0])
    if train_or_test == 'train':
        line = '%s ' % labels[i]
    else:
        line = ''
    non_zero_values_line_i = list(map(lambda v: '%s:%s:%s' % (feature_range[v[0]], v[0], int(v[1])), non_zero_values_line_i))
    line += ' '.join(non_zero_values_line_i)
    line += '\n'
    f_libffm.write(line)

f_libffm.close()
sys.exit(0)
