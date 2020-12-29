#%%

import collections
import json
import numpy as np
import os
import sys
sys.path.append("ecg")
import scipy.stats as sst
from keras.layers import Layer
from keras import backend as K
import keras
import glob

import util
import load

#%%
data_path = "./examples/cinc17/dev.json"

# gpu-1 adam0.001 reg0.001 a1.0
model_folder_path = "./saved_res_nobn/cinc17"
arr = os.listdir(model_folder_path)
arr = sorted(arr)
last_folder = arr[-1]
model_folder_path = "{}/{}/*.hdf5".format(model_folder_path, last_folder)
arr_file = glob.glob(model_folder_path)
print('arr_file', arr_file)
file_name = arr_file[0]
model_path = file_name
print('Model Path : ', model_path)
# exit()
# model_path = "../../../saved_res_nobn/cinc17/1609222106-676/14.899-0.302-001-16.664-0.284.hdf5"

data = load.load_dataset(data_path)
preproc = util.load(os.path.dirname(model_path))
print('preproc window size : ', preproc.window_size)

class ScaleLayer(Layer):
    def __init__(self, alpha=0):
      super(ScaleLayer, self).__init__()
      self.alpha = alpha
      self.scale = K.variable(self.alpha, dtype='float32', name='alpha')

    def get_config(self):
      return {"alpha": self.alpha}

    def call(self, inputs):
      return inputs * self.scale

# load model
# model = load_model(model_path, custom_objects={'ScaleLayer':ScaleLayer})
model = keras.models.load_model(model_path, custom_objects={'ScaleLayer':ScaleLayer})

#%%
data_path = "./examples/cinc17/train.json"
with open("./examples/cinc17/train.json", 'rb') as fid:
    train_labels = [json.loads(l)['labels'] for l in fid]
counts = collections.Counter(preproc.class_to_int[l[0]] for l in train_labels)
counts = sorted(counts.most_common(), key=lambda x: x[0])
counts = list(zip(*counts))[1]
smooth = 500
counts = np.array(counts)[None, None, :]
total = np.sum(counts) + counts.shape[1]
prior = (counts + smooth) / float(total)
##%
print(prior)

#%%

probs = []
labels = []
for x, y  in zip(*data):
    x, y = preproc.process([x], [y])
    probs.append(model.predict(x))
    # print(sst.mode(np.argmax(probs[0] / prior, axis=2).squeeze()))
    labels.append(y)

#%%

preds = []
ground_truth = []
preds_prior = []
for p, g in zip(probs, labels):
    preds.append(sst.mode(np.argmax(p, axis=2).squeeze())[0][0])
    preds_prior.append(sst.mode(np.argmax(p / prior, axis=2).squeeze())[0][0])
    ground_truth.append(sst.mode(np.argmax(g, axis=2).squeeze())[0][0])

#%%

# gt = np.array(ground_truth)
# print('0:', len(gt[gt == 0]))
# print('1:', len(gt[gt == 1]))
# print('2:', len(gt[gt == 2]))
# print('3:', len(gt[gt == 3]))
# print(preds)

#%%

import sklearn.metrics as skm
report = skm.classification_report(
            ground_truth, preds,
            target_names=preproc.classes,
            digits=3)
scores = skm.precision_recall_fscore_support(
                    ground_truth,
                    preds,
                    average=None)
print(report)
print("CINC Average {:3f}".format(np.mean(scores[2][:3])))
report = skm.classification_report(
            ground_truth, preds_prior,
            target_names=['A', 'N', 'O', '~'],
            digits=3)

scores = skm.precision_recall_fscore_support(
                    ground_truth,
                    preds_prior,
                    average=None)
print('report w/ prior')
print(report)
print("CINC Average {:3f}".format(np.mean(scores[2][:3])))
#%%


