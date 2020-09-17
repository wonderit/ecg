from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import json
import keras
import numpy as np
import os
import random
import time

import network
import load
import util

MAX_EPOCHS = 100

def make_save_dir(dirname, experiment_name):
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def get_filename_for_saving(save_dir):
    return os.path.join(save_dir,
            "{val_loss:.3f}-{val_acc:.3f}-{epoch:03d}-{loss:.3f}-{acc:.3f}.hdf5")

def train(args, params):

    print("Loading training set...")
    train = load.load_dataset(params['train'])
    print("Loading dev set...")
    dev = load.load_dataset(params['dev'])
    print("Building preprocessor...")
    preproc = load.Preproc(*train)
    print("Training size: " + str(len(train[0])) + " examples.")
    print("Dev size: " + str(len(dev[0])) + " examples.")


    save_dir = make_save_dir(params['save_dir'], args.experiment)

    util.save(preproc, save_dir)

    print('classes', preproc.classes)

    params.update({
        "input_shape": [None, 1],
        "num_categories": len(preproc.classes)
    })

    model = network.build_network(**params)

    stopping = keras.callbacks.EarlyStopping(patience=8)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=params["learning_rate"] * 0.001)

    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=get_filename_for_saving(save_dir),
        save_best_only=False)

    batch_size = params.get("batch_size", 32)

    if params.get("generator", False):
        train_gen = load.data_generator(batch_size, preproc, *train)
        dev_gen = load.data_generator(batch_size, preproc, *dev)

        train_gen = np.array(train_gen)
        dev_gen = np.array(dev_gen)
        # print('train_gen m, s: ', train_gen.mean(), train_gen.std())
        # print('train_gen min, max: ', train_gen.min(), train_gen.max())

        print('train_gen shape : ', train_gen.shape)
        print('dev_gen shape : ', train_gen.shape)

        model.fit_generator(
            train_gen,
            steps_per_epoch=int(len(train[0]) / batch_size),
            epochs=MAX_EPOCHS,
            validation_data=dev_gen,
            validation_steps=int(len(dev[0]) / batch_size),
            callbacks=[checkpointer, reduce_lr, stopping])
    else:
        train_x, train_y = preproc.process(*train)
        dev_x, dev_y = preproc.process(*dev)

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(dev_x)
        test_y = np.array(dev_y)

        # import matplotlib.pyplot as plt
        # for i in range(100):
        #     plt.plot(train_x[0 + i, :1024, 0])
        #
        # plt.savefig('MLII_0_original.png', edgecolor='black', dpi=600)
        print(train_x.shape, test_x.shape)
        print(train_y.shape, test_y.shape)
        random_train_x = []
        random_train_y = []
        window_size = 256
        # print(train_x[10, 7*window_size:8*window_size, :])
        print(train_y[10, 7, :])

        import matplotlib.pyplot as plt
        for i in range(40):
            # plt.plot(train_x[0 + i, :1024, 0])
            plt.plot(train_x[10, i * window_size:(i+1) * window_size, :])
        plt.savefig('MLII_10_original.png', edgecolor='black', dpi=600)

        plt.clf()
        plt.plot(np.argmax(train_y[10, :, :], axis=1))
        plt.savefig('10_class.png', edgecolor='black', dpi=600)
        # exit()
        # get random 1 window for each patient
        # loop through patients
        # for i in range(train_x.shape[0]):
        #     # get random index window
        #     tr_i = np.random.choice(train_x.shape[1] // window_size, 1)[0]
        #     random_train_x.append(train_x[i, tr_i*window_size:(tr_i+1)*window_size, :])
        #     random_train_y.append(train_y[i, tr_i, :])


        all_train_x = []
        all_train_y = []
        n_sample = 1
        for i in range(train_x.shape[0]):
            # get random index window for n times

            for j in range(n_sample):
                # tr_i = np.random.choice(train_x.shape[1] // window_size, 1)[0]
                tr_i = j
                all_train_x.append(train_x[i, tr_i*window_size:(tr_i+1)*window_size, :])
                all_train_y.append(train_y[i, tr_i, :])

        # random_test_x = []
        # random_test_y = []
        # get random 1 window for each patient
        # loop through patients
        # for j in range(test_x.shape[0]):
        #     # get random index window
        #     te_i = np.random.choice(test_x.shape[1] // window_size, 1)[0]
        #     random_test_x.append(test_x[j, te_i * window_size:(te_i + 1) * window_size, :])
        #     random_test_y.append(test_y[j, te_i, :])
        # r_i = np.random.choice(test_x.shape[1] // window_size, 1)[0]
        r_i = 0
        r_length = 1
        # print('random number : ', np.random.choice(test_x.shape[1] // window_size, 1)[0])
        # train_x = train_x[:, 256 * r_i:256 * (r_i+1)* r_length, :]
        # train_y = train_y[:, r_i, :]

        test_x = test_x[:, 256 * r_i:256 * (r_i+1) * r_length, :]
        test_y = test_y[:, r_i, :]
        train_x = np.array(all_train_x)
        train_y = np.array(all_train_y)

        # test_x = np.array(random_test_x)
        # train_x = np.array(train_x)

        # train_x = np.squeeze(train_x, axis=(2,))
        # test_x = np.squeeze(test_x, axis=(2,))

        print('train_x m, s: ', train_x.mean(), train_x.std())
        print('train_x min, max: ', train_x.min(), train_x.max())

        print('train_x shape : ', train_x.shape)
        print('train_y shape : ', train_y.shape)
        print('test_x shape : ', test_x.shape)
        print('test_y shape : ', test_y.shape)
        gt = np.argmax(test_y, axis=1)
        print('0:', len(gt[gt == 0]))
        print('1:', len(gt[gt == 1]))
        print('2:', len(gt[gt == 2]))
        print('3:', len(gt[gt == 3]))
        print('gt shape:', gt.shape)

        data_dir = 'processed_data'

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        train_file_suffix = 'train'
        test_file_suffix = 'test'

        file_name_train_x = 'X{}'.format(train_file_suffix)
        file_name_train_y = 'y{}'.format(train_file_suffix)
        file_name_test_x = 'X{}'.format(test_file_suffix)
        file_name_test_y = 'y{}'.format(test_file_suffix)

        # np.savetxt('{}/{}'.format(data_dir, file_name_train_x), train_x, delimiter=',', fmt='%1.8f')
        # np.savetxt('{}/{}'.format(data_dir, file_name_train_y), train_y, delimiter=',', fmt='%1.8f')
        # np.savetxt('{}/{}'.format(data_dir, file_name_test_x), test_x, delimiter=',', fmt='%1.8f')
        # np.savetxt('{}/{}'.format(data_dir, file_name_test_y), test_y, delimiter=',', fmt='%1.8f')
        # exit()

        model.fit(
            train_x, train_y,
            batch_size=batch_size,
            epochs=MAX_EPOCHS,
            validation_data=(dev_x, dev_y),
            callbacks=[checkpointer, reduce_lr, stopping])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--experiment", "-e", help="tag with experiment name",
                        default="default")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    train(args, params)
