import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

train_x = np.genfromtxt('../../processed_data/Xtrain', delimiter=',', dtype='float')
train_y = np.genfromtxt('../../processed_data/ytrain', delimiter=',', dtype='float')

test_x = np.genfromtxt('../../processed_data/Xtest', delimiter=',', dtype='float')
test_y = np.genfromtxt('../../processed_data/ytest', delimiter=',', dtype='float')

print('Data Loading finished (row:{})'.format(len(train_x)))

batch_size=32
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.kernel_size = 7
        self.padding_size = 0
        self.channel_size = 16
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avgpool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avgpool4 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avgpool5 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(1, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv3 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv4 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv5 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32
        x = self.avgpool1(x)  # 32
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.avgpool3(x)
        x = F.relu(self.conv4(x))
        x = self.avgpool4(x)
        x = F.relu(self.conv5(x))
        x = self.avgpool5(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y

class NetMaxpool(nn.Module):
    def __init__(self):
        super(NetMaxpool, self).__init__()
        self.kernel_size = 7
        self.padding_size = 0
        self.channel_size = 16
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.maxpool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(1, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv3 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv4 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        # self.conv5 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
        #                        padding=(self.kernel_size // 2))
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32
        x = self.maxpool1(x)  # 32
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = F.relu(self.conv4(x))
        x = self.maxpool4(x)
        # x = F.relu(self.conv5(x))
        # x = self.maxpool5(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y

model = NetMaxpool()


class ECGDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)


train_dataset = ECGDataset(train_x, train_y)
test_dataset = ECGDataset(test_x, test_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)

val_x = torch.from_numpy(test_x).float()
val_y = torch.from_numpy(test_y).float()

def train(epoch):
    tr_loss = 0.0
    val_loss = 0.0
    model.train()
    for batch_idx, data in enumerate(train_loader):
        # get the inputs
        tr_inputs, tr_labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # unsqueeze data
        tr_inputs = tr_inputs.unsqueeze(1)

        # one-hot to label
        tr_labels = torch.argmax(tr_labels, dim=1)
        # tr_labels = Variable(tr_labels)

        # get the validation set
        x_val, y_val = Variable(val_x), Variable(val_y)
        val_inputs = x_val.unsqueeze(1)
        val_labels = torch.argmax(y_val, dim=1)


        # forward + backward + optimize
        tr_outputs = model(tr_inputs)
        val_outputs = model(val_inputs)

        # softmax
        # print('tr_outputs', tr_outputs.shape)
        # print('tr_outputs before softmax', tr_outputs[0])
        # tr_outputs = torch.softmax(tr_outputs, dim=-1)
        # print('tr_outputs after softmax', tr_outputs[0])
        # val_outputs = torch.softmax(val_outputs, dim=-1)

        # loss
        loss_train = criterion(tr_outputs, tr_labels)
        loss_val = criterion(val_outputs, val_labels)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()

        batch_tr_loss = loss_train.detach().item()
        batch_val_loss = loss_val.detach().item()
        train_losses.append(batch_tr_loss)
        val_losses.append(batch_val_loss)

        # print statistics
        tr_loss += batch_tr_loss
        val_loss += batch_val_loss
        if batch_idx % 100 == 99:  # print every 2000 mini-batches
            print('[%d, %5d] train loss: %.6f, val loss : %.6f' %
                  (epoch + 1, batch_idx + 1, tr_loss / 100, val_loss / 100))
            tr_loss = 0.0
            val_loss = 0.0


# defining the number of epochs
n_epochs = 10
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# training the model
for epoch in range(n_epochs):
    train(epoch)

print('Finish Training')

# plotting the training and validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
# plt.show()
plt.savefig('cinc17_learning_curve_txt.png', edgecolor='black', dpi=600)

val_inputs = val_x.unsqueeze(1)
val_labels = torch.argmax(val_y, dim=1)
print('0:', len(val_labels[val_labels == 0]))
print('1:', len(val_labels[val_labels == 1]))
print('2:', len(val_labels[val_labels == 2]))
print('3:', len(val_labels[val_labels == 3]))
ground_truth = val_labels

# #  Evaluation
# model_path = "../../../saved/cinc17/1597729558-4/0.415-0.863-017-0.264-0.910.hdf5"
# data_path = "../dev.json"
# import sys
# sys.path.append("../../../ecg")
# import scipy.stats as sst
#
# import load
# import util
#
# data = load.load_dataset(data_path)
# preproc = util.load(os.path.dirname(model_path))
#
# prior = [[[0.15448743, 0.66301941, 0.34596848, 0.09691286]]]
#
# probs = []
# labels = []
# for x, y  in zip(*data):
#     x, y = preproc.process([x], [y])
#     print(x.shape)
#     print(y.shape)
#     exit()
#     probs.append(model.predict(x))
#     print(sst.mode(np.argmax(probs[0] / prior, axis=2).squeeze()))
#     break
#     labels.append(y)

preds = torch.argmax(model(val_inputs), dim=1)

import sklearn.metrics as skm
report = skm.classification_report(
            ground_truth, preds,
            target_names=['A', 'N', 'O', '~'],
            digits=3)
scores = skm.precision_recall_fscore_support(
                    ground_truth,
                    preds,
                    average=None)
print(report)