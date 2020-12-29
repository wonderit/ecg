#%%

import collections
import json
import keras
import numpy as np
import os
import sys
sys.path.append("../../../ecg")
import scipy.stats as sst
from keras.layers import Layer
import keras as K
from tensorflow.keras.models import load_model

import util
import load

#%%
data_path = "../dev.json"

# gpu-1 adam0.001 reg0.001 a1.0
model_path = "../../../saved_res_nobn/cinc17/16092221230-792/17.155-0.189-001-18.789-0.177.hdf5"

data = load.load_dataset(data_path)
preproc = util.load(os.path.dirname(model_path))
print('preproc window size : ', preproc.window_size)

class ScaleLayer(Layer):
    def __init__(self, alpha=0):
      super(ScaleLayer, self).__init__()
      self.alpha = alpha
      self.scale = K.variable(self.alpha, dtype='float32', name='alpha')

    def get_config(self):
      # config = super().get_config()
      # config["alpha"] = self.alpha
      return {"alpha": self.alpha}

    def call(self, inputs):
      return inputs * self.scale

# load model
model = load_model(model_path, custom_objects={'ScaleLayer':ScaleLayer})
# model = keras.models.load_model(model_path, custom_objects={'ScaleLayer':ScaleLayer})

#%%

data_path = "../train.json"
with open("../train.json", 'rb') as fid:
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


