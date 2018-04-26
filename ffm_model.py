import numpy as np
import xlearn as xl
from sklearn.externals import joblib
import sys

base_dir = './data/'

train_file, eval_file, test_file = sys.argv[1], sys.argv[2], sys.argv[3]
train_file_part_num = train_file.split('.')[0][-1]
model_path = base_dir+'ffm_model_part_%s.bin' % train_file_part_num

param = {'task': 'binary', 'lr': 0.05, 'epoch': 10, 'lambda': 0.002, 'metric': 'auc'}

ffm_model = xl.create_ffm()
ffm_model.setTrain(train_file)
ffm_model.setValidate(eval_file)

ffm_model.fit(param, model_path)

ffm_model.setTest(eval_file)
ffm_model.setSigmoid()
ffm_model.predict(model_path, eval_file.split('.')[0]+'.res')

ffm_model.setTest(test_file)
ffm_model.setSigmoid()
ffm_model.predict(model_path, test_file.split('.')[0]+'_%s.res' % train_file_part_num)