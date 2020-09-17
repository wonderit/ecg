import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
torch.manual_seed(0)
data_dir = 'minimum_data'

train_x = np.genfromtxt('../../{}/Xtrain'.format(data_dir), delimiter=',', dtype='float')
train_y = np.genfromtxt('../../{}/ytrain'.format(data_dir), delimiter=',', dtype='float')

test_x = np.genfromtxt('../../{}/Xtest'.format(data_dir), delimiter=',', dtype='float')
test_y = np.genfromtxt('../../{}/ytest'.format(data_dir), delimiter=',', dtype='float')

print('Data Loading finished (row:{})'.format(len(train_x)))

batch_size = 32
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
        # self.maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.maxpool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(1, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv11 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv22 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv3 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        self.conv33 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=(self.kernel_size // 2))
        # self.conv4 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
        #                        padding=(self.kernel_size // 2))
        # self.conv5 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
        #                        padding=(self.kernel_size // 2))
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32
        x = F.relu(self.conv11(x))  # 32
        x = self.dropout(x)
        x = self.maxpool1(x)  # 32
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv22(x))
        x = self.dropout(x)
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv33(x))
        x = self.dropout(x)
        x = self.maxpool3(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.dropout(x)
        # x = self.maxpool4(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        y = self.fc3(x)
        return y



class ML4CVD_shallow(nn.Module):
    def __init__(self):
        super(ML4CVD_shallow, self).__init__()
        self.kernel_size = 7
        self.channel_size = 8
        self.conv1 = nn.Conv1d(1, self.channel_size, kernel_size=self.kernel_size, padding=(self.kernel_size // 2))
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size, padding=(self.kernel_size // 2))
        self.conv3 = nn.Conv1d(self.channel_size * 2, self.channel_size, kernel_size=self.kernel_size, padding=(self.kernel_size // 2))
        self.conv4 = nn.Conv1d(self.channel_size * 3, 24, kernel_size=self.kernel_size, padding=(self.kernel_size // 2))
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(24, 24, kernel_size=self.kernel_size, padding=(self.kernel_size // 2))
        self.conv6 = nn.Conv1d(48, 24, kernel_size=self.kernel_size, padding=(self.kernel_size // 2))
        self.conv7 = nn.Conv1d(72, 16, kernel_size=self.kernel_size, padding=(self.kernel_size // 2))
        self.conv8 = nn.Conv1d(16, 16, kernel_size=self.kernel_size, padding=(self.kernel_size // 2))
        self.conv9 = nn.Conv1d(32, 16, kernel_size=self.kernel_size, padding=(self.kernel_size // 2))
        self.fc1 = nn.Linear(3072, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 4)
        # self.fc1 = nn.Linear(5620, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 32
        x = F.relu(self.conv2(x)) # 32
        x = self.avgpool1(x) # 32
        x1 = F.relu(self.conv2(x))
        c1 = torch.cat((x, x1), dim=1) # 64
        x2 = F.relu(self.conv3(c1)) # 32
        y = torch.cat((x, x1, x2), dim=1) # 96
        # downsizing
        y = F.relu(self.conv4(y)) # 24
        y = self.avgpool1(y)

        x3 = F.relu(self.conv5(y))
        c2 = torch.cat((y, x3), dim=1)
        x4 = F.relu(self.conv6(c2))
        y = torch.cat((y, x3, x4), dim=1)

        y = F.relu(self.conv7(y))
        y = self.avgpool1(y)

        x5 = F.relu(self.conv8(y))
        c3 = torch.cat((y, x5), dim=1)
        x6 = F.relu(self.conv9(c3))
        y = torch.cat((y, x5, x6), dim=1)

        # Flatten
        y = y.view(y.size(0), -1)

        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        return y


class ML4CVD(nn.Module):
    def __init__(self):
        super(ML4CVD, self).__init__()
        self.kernel_size = 71
        self.padding_size = 35
        self.channel_size = 32
        self.conv1 = nn.Conv1d(1, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=self.padding_size)
        self.conv22 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                                padding=self.padding_size)
        self.conv3 = nn.Conv1d(self.channel_size * 2, self.channel_size, kernel_size=self.kernel_size,
                               padding=self.padding_size)
        self.conv4 = nn.Conv1d(self.channel_size * 3, 24, kernel_size=self.kernel_size, padding=self.padding_size)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(24, 24, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv6 = nn.Conv1d(48, 24, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv7 = nn.Conv1d(72, 16, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv8 = nn.Conv1d(16, 16, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv9 = nn.Conv1d(32, 16, kernel_size=self.kernel_size, padding=self.padding_size)
        self.fc1 = nn.Linear(12288, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 4)
        # self.fc1 = nn.Linear(5620, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32
        x = F.relu(self.conv2(x))  # 32
        x = self.avgpool1(x)  # 32
        x1 = F.relu(self.conv22(x))
        c1 = torch.cat((x, x1), dim=1)  # 64
        x2 = F.relu(self.conv3(c1))  # 32
        y = torch.cat((x, x1, x2), dim=1)  # 96
        # downsizing
        y = F.relu(self.conv4(y))  # 24
        y = self.avgpool1(y)

        x3 = F.relu(self.conv5(y))
        c2 = torch.cat((y, x3), dim=1)
        x4 = F.relu(self.conv6(c2))
        y = torch.cat((y, x3, x4), dim=1)

        y = F.relu(self.conv7(y))
        y = self.avgpool1(y)

        x5 = F.relu(self.conv8(y))
        c3 = torch.cat((y, x5), dim=1)
        x6 = F.relu(self.conv9(c3))
        y = torch.cat((y, x5, x6), dim=1)

        # Flatten
        y = y.view(y.size(0), -1)

        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        return y

model = NetMaxpool()
# model = ML4CVD_shallow()
from torchsummary import summary
summary(model, input_size =(1, 512), batch_size=32)


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

        model.train()
        # forward + backward + optimize
        tr_outputs = model(tr_inputs)

        model.eval()
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
n_epochs = 30
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

prior = [0.15448743, 0.66301941, 0.34596848, 0.09691286]


# using prior

probs = []
probs_prior = []
model.eval()
val_outputs = model(val_inputs)
np_outputs = val_outputs.detach().numpy()
from scipy.special import softmax
for output_array in np_outputs:
    ss = softmax(output_array)
    ss2 = softmax(output_array) / np.array(prior)
    probs.append(np.argmax(ss))
    probs_prior.append(np.argmax(ss2))

preds = np.array(probs)
preds_prior = np.array(probs_prior)

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


report = skm.classification_report(
            ground_truth, preds_prior,
            target_names=['A', 'N', 'O', '~'],
            digits=3)
print('report w/ prior')
print(report)