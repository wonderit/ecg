#%%

import collections
import json
import keras
import numpy as np
import os
import sys
sys.path.append("../../../ecg")
import scipy.stats as sst

import util
import load

#%%
# window size : 71 - w/BN - Residual
# model_path = "../../../saved/cinc17/1600732571-86/0.226-0.922-020-0.131-0.954.hdf5"
# window size : 71 - w/BN
# model_path = "../../../saved/cinc17/1600729557-777/0.251-0.910-018-0.219-0.921.hdf5"
# window size : 71 : showed poor performance -> trained twice
# model_path = "../../../saved/cinc17/1600731414-688/0.267-0.905-018-0.201-0.927.hdf5"
# model_path = "../../../saved/cinc17/1600725047-904/0.279-0.899-020-0.210-0.923.hdf5"
# window size : 60
# model_path = "../../../saved/cinc17/1600723971-398/0.291-0.897-018-0.228-0.918.hdf5"
# window size : 50
# model_path = "../../../saved/cinc17/1600723290-689/0.422-0.854-020-0.297-0.893.hdf5"
# window size : 40
# model_path = "../../../saved/cinc17/1600722878-856/0.414-0.859-020-0.271-0.903.hdf5"
# window size : 30
# model_path = "../../../saved/cinc17/1600721655-973/0.528-0.824-020-0.280-0.899.hdf5"
# window size : 20
# model_path = "../../../saved/cinc17/1600721125-894/0.535-0.805-018-0.363-0.866.hdf5"
# window size : 10
# model_path = "../../../saved/cinc17/1600720547-201/0.571-0.789-021-0.454-0.830.hdf5"

# W/BN start
data_path = "../dev.json"
# ws : 60
model_path = "../../../saved/cinc17/1600748105-72/0.284-0.900-018-0.224-0.920.hdf5"
# ws : 50
model_path = "../../../saved/cinc17/1600749241-120/0.327-0.884-018-0.264-0.904.hdf5"
# ws : 40
model_path = "../../../saved/cinc17/1600752602-261/0.389-0.866-018-0.265-0.906.hdf5"
# ws : 30
model_path = "../../../saved/cinc17/1600755293-457/0.440-0.838-018-0.370-0.867.hdf5"
# ws : 20
model_path = "../../../saved/cinc17/1600755766-211/0.481-0.823-015-0.415-0.850.hdf5"
# ws : 10
model_path = "../../../saved/cinc17/1600756084-472/0.592-0.786-018-0.480-0.816.hdf5"

data = load.load_dataset(data_path)
preproc = util.load(os.path.dirname(model_path))
model = keras.models.load_model(model_path)

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


