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
# model_path = "../../../saved/cinc17/1600748105-72/0.284-0.900-018-0.224-0.920.hdf5"
# # ws : 50
# model_path = "../../../saved/cinc17/1600749241-120/0.327-0.884-018-0.264-0.904.hdf5"
# # ws : 40
# model_path = "../../../saved/cinc17/1600752602-261/0.389-0.866-018-0.265-0.906.hdf5"
# # ws : 30
# model_path = "../../../saved/cinc17/1600755293-457/0.440-0.838-018-0.370-0.867.hdf5"
# # ws : 20
# model_path = "../../../saved/cinc17/1600755766-211/0.481-0.823-015-0.415-0.850.hdf5"
# # ws : 10
# model_path = "../../../saved/cinc17/1600756084-472/0.592-0.786-018-0.480-0.816.hdf5"
# w/o batch norm start
# ws : 10
model_path = "../../../saved/cinc17/1600756367-348/0.727-0.753-018-0.443-0.837.hdf5"
# ws : 20
model_path = "../../../saved/cinc17/1600756668-322/0.593-0.806-018-0.312-0.889.hdf5"
# ws : 30
model_path = "../../../saved/cinc17/1600756966-153/0.465-0.836-018-0.335-0.881.hdf5"
# ws : 40
model_path = "../../../saved/cinc17/1600757298-81/0.346-0.883-018-0.186-0.934.hdf5"
# residual model
# ws : 10
# model_path = "../../../saved/cinc17/1600783803-90/0.588-0.780-018-0.511-0.800.hdf5"
# # ws 20
# model_path = "../../../saved/cinc17/1600785113-586/0.459-0.836-018-0.315-0.885.hdf5"
# # ws 30
# model_path = "../../../saved/cinc17/1600787161-453/0.373-0.868-018-0.309-0.890.hdf5"
# # ws 40
# model_path = "../../../saved/cinc17/1600788639-728/0.320-0.885-018-0.251-0.910.hdf5"
# # ws 50
# model_path = "../../../saved/cinc17/1600790298-425/0.284-0.901-018-0.202-0.930.hdf5"
# # ws 60
# model_path = "../../../saved/cinc17/1600792595-994/0.247-0.912-017-0.196-0.929.hdf5"
# regular w/o bn, w/ dropout 0.2, ws : 30
model_path = "../../../saved/cinc17/1600871065-652/0.571-0.760-018-0.556-0.770.hdf5"
model_path = "../../../saved/cinc17/1600871065-652/0.571-0.759-026-0.555-0.773.hdf5"
# regular w/o bn, w/ dropout 0.2, ws : 71
model_path = "../../../saved/cinc17/1600871658-787/0.307-0.887-025-0.279-0.898.hdf5"
# regular w/o bn, w/ dropout 0.2, ws : 10
model_path = "../../../saved/cinc17/1600873097-592/0.883-0.654-024-0.775-0.685.hdf5"
# regular w/o bn, w/ dropout 0.2, ws : 20
model_path = "../../../saved/cinc17/1600873600-115/0.737-0.699-030-0.688-0.727.hdf5"
# regular w/o bn, w/ dropout 0.2, ws : 40
model_path = "../../../saved/cinc17/1600873962-480/0.436-0.848-028-0.407-0.855.hdf5"
# regular w/o bn, w/ dropout 0.2, ws : 50
model_path = "../../../saved/cinc17/1600874444-895/0.403-0.854-020-0.369-0.866.hdf5"
# regular w/o bn, w/ dropout 0.2, ws : 60
model_path = "../../../saved/cinc17/1600876556-803/0.308-0.890-020-0.281-0.897.hdf5"

# regular w bn, w dropout 0.2 ws : 71
model_path = "../../../saved/cinc17/1600887040-261/0.260-0.908-017-0.247-0.912.hdf5"
#60
model_path = "../../../saved/cinc17/1600888274-270/0.302-0.895-024-0.293-0.895.hdf5"
# 50
model_path = "../../../saved/cinc17/1600890816-290/0.334-0.882-018-0.337-0.877.hdf5"
# 40
model_path = "../../../saved/cinc17/1600893445-585/0.387-0.861-020-0.370-0.867.hdf5"
# 30
model_path = "../../../saved/cinc17/1600894767-800/0.431-0.845-020-0.423-0.847.hdf5"
# 20
model_path = "../../../saved/cinc17/1600896351-369/0.511-0.819-020-0.499-0.815.hdf5"
# 10
model_path = "../../../saved/cinc17/1600898469-66/0.693-0.725-030-0.648-0.738.hdf5"

# w/o dropout, w/ bn resnet ws : 10
model_path = "../../../saved/cinc17/1600969139-916/0.555-0.794-020-0.427-0.840.hdf5"

# w/ dropout, no bn resnet ws : 10
model_path = "../../../saved/cinc17/1600969139-916/0.555-0.794-020-0.427-0.840.hdf5"

# of layer : 16 -> 32 ws : 10 1600977494-766 4s
model_path = "../../../saved/cinc17/1600977494-766/0.707-0.757-018-0.413-0.846.hdf5"
# of layer : 16 -> 32 ws : 20 6s
model_path = "../../../saved/cinc17/1600977675-274/0.595-0.796-018-0.385-0.860.hdf5"
# of layer : 16 -> 32 ws : 30 9s lr : 0.0005
model_path = "../../../saved/cinc17/1600977811-169/0.488-0.832-020-0.289-0.897.hdf5"
# of layer : 16 -> 32 ws : 40 10s
model_path = "../../../saved/cinc17/1600978066-910/0.437-0.854-020-0.244-0.916.hdf5"
# of layer : 16 -> 32 ws : 50 12 s
model_path = "../../../saved/cinc17/1600978549-367/0.359-0.876-020-0.246-0.912.hdf5"
# of layer : 16 -> 32 ws : 60 15 s
model_path = "../../../saved/cinc17/1600979318-106/0.302-0.900-020-0.211-0.925.hdf5"
# of layer : 16 -> 32 ws : 71 18 s
model_path = "../../../saved/cinc17/1600979659-741/0.279-0.909-025-0.170-0.939.hdf5"

# resnet bn, drop0 ws : 40 73 s
model_path = "../../../saved_res_nobn/cinc17/1607990251-512/0.324-0.885-020-0.193-0.932.hdf5"

data = load.load_dataset(data_path)
preproc = util.load(os.path.dirname(model_path))
print('preproc window size : ', preproc.window_size)
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


