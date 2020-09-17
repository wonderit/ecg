import matplotlib.pyplot as plt
import numpy as np
from biosppy.signals import tools as st

X_train = np.genfromtxt('../../processed_data/Xtrain', delimiter=',', dtype='float')
y_train = np.genfromtxt('../../processed_data/ytrain', delimiter=',', dtype='float')

X_test = np.genfromtxt('../../processed_data/Xtest', delimiter=',', dtype='float')
y_test = np.genfromtxt('../../processed_data/ytest', delimiter=',', dtype='float')

print('Data Loading finished (row:{})'.format(len(X_train)))

def scale_maxabs(arr, maxabs, thres):
    arr = (arr / maxabs) * thres
    return arr

def apply_threshold(arr, thres):
    arr[arr > thres] = thres
    arr[arr < -thres] = thres
    return arr

def compute_mean_std(x):
    x = np.hstack(x)
    return (np.mean(x).astype(np.float32),
           np.std(x).astype(np.float32))

# X_train = X_train.reshape([7676, 1, 256, 8]).mean(3).mean(1)
# X_test = X_test.reshape([852, 1, 256, 8]).mean(3).mean(1)
X_train = X_train.reshape([7676, 1, 512, 2560 // 512]).mean(3).mean(1)
X_test = X_test.reshape([852, 1, 512, 2560 // 512]).mean(3).mean(1)

order = int(0.3 * 100)
filtered_train_x, _, _ = st.filter_signal(signal=X_train,
                                          ftype='FIR',
                                          band='bandpass',
                                          order=order,
                                          frequency=[3, 45],
                                          sampling_rate=100)
filtered_test_x, _, _ = st.filter_signal(signal=X_test,
                                          ftype='FIR',
                                          band='bandpass',
                                          order=order,
                                          frequency=[3, 45],
                                          sampling_rate=100)

mean, std = compute_mean_std(filtered_train_x)
print(mean, std)

# X_threshold = apply_threshold(filtered_train_x, 4000)
# test_x_thres = apply_threshold(filtered_test_x, 4000)
# current_x = scale_maxabs(X_threshold, np.max(np.abs(X_threshold)), 30)
# current_test_x = scale_maxabs(test_x_thres, np.max(np.abs(test_x_thres)), 30)

current_x = (filtered_train_x - mean) / std
current_test_x = (filtered_test_x - mean) / std
import os
data_dir = '../../minimum_data'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
train_file_suffix = 'train'
test_file_suffix = 'test'

file_name_train_x = 'X{}'.format(train_file_suffix)
file_name_train_y = 'y{}'.format(train_file_suffix)
file_name_test_x = 'X{}'.format(test_file_suffix)
file_name_test_y = 'y{}'.format(test_file_suffix)

np.savetxt('{}/{}'.format(data_dir, file_name_train_x), current_x, delimiter=',', fmt='%1.8f')
np.savetxt('{}/{}'.format(data_dir, file_name_train_y), y_train, delimiter=',', fmt='%1.8f')
np.savetxt('{}/{}'.format(data_dir, file_name_test_x), current_test_x, delimiter=',', fmt='%1.8f')
np.savetxt('{}/{}'.format(data_dir, file_name_test_y), y_test, delimiter=',', fmt='%1.8f')
