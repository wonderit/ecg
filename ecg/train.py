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

from keras.utils import plot_model

# MAX_EPOCHS = 100
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def make_save_dir(dirname, experiment_name):
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def get_filename_for_saving(save_dir):
    return os.path.join(save_dir,
            "{val_loss:.3f}-{val_accuracy:.3f}-{epoch:03d}-{loss:.3f}-{accuracy:.3f}.hdf5")

def get_filename_json_for_saving(save_dir):
    return os.path.join(save_dir,
            "{val_loss:.3f}-{val_accuracy:.3f}-{epoch:03d}-{loss:.3f}-{accuracy:.3f}.json")

def train(args, params):

    print("Loading training set...")
    train = load.load_dataset(params['train'])
    print("Loading dev set...")
    dev = load.load_dataset(params['dev'])
    print("Building preprocessor...")
    preproc = load.Preproc(*train)
    print("Training size: " + str(len(train[0])) + " examples.")
    print("Dev size: " + str(len(dev[0])) + " examples.")

    preproc.set_window_size(params['window_size'])

    save_dir = make_save_dir(params['save_dir'], args.experiment)

    util.save(preproc, save_dir)

    print('classes', preproc.classes)

    params.update({
        "input_shape": [None, 1],
        "num_categories": len(preproc.classes)
    })

    model = network.build_network(**params)
    print(model.summary())
    if params.get('is_regular_conv', False):
        plot_model(model, to_file='model_regular_conv.png', show_shapes=True)
    else:
        if params.get('conv_batch_norm'):
            plot_model(model, to_file='model_residual_conv_bn.png', show_shapes=True)
        else:
            plot_model(model, to_file='model_residual_conv_nobn.png', show_shapes=True)

    stopping = keras.callbacks.EarlyStopping(patience=3)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=params["learning_rate"] * 0.001)

    # checkpointer = keras.callbacks.ModelCheckpoint(
    #     filepath=get_filename_for_saving(save_dir),
    #     save_best_only=False, save_weights_only=False)
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=get_filename_for_saving(save_dir),
        save_weights_only=True,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
    )

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
            epochs=params.get("epoch", 10),
            validation_data=dev_gen,
            validation_steps=int(len(dev[0]) / batch_size),
            callbacks=[checkpointer, reduce_lr, stopping])
    else:
        train_x, train_y = preproc.process(*train)
        dev_x, dev_y = preproc.process(*dev)

        print('train_x original shape : ', train_x.shape)
        print('train_y original shape : ', train_y.shape)
        print('test_x original shape : ', dev_x.shape)
        print('test_y original shape : ', dev_y.shape)

        # train_x = np.array(train_x)
        # train_y = np.array(train_y)
        # test_x = np.array(dev_x)
        # test_y = np.array(dev_y)

        window_size = 256
        n_sample = 40

        r_i = 0
        r_length = 1
        # print('random number : ', np.random.choice(test_x.shape[1] // window_size, 1)[0])
        # train_x = train_x[:, 256 * r_i:256 * (r_i+1)* r_length, :]
        # train_y = train_y[:, r_i, :]
        #
        # test_x = test_x[:, 256 * r_i:256 * (r_i+1) * r_length, :]
        # test_y = test_y[:, r_i, :]
        # train_x = np.array(all_train_x)
        # train_y = np.array(all_train_y)

        # FOR PLAINTEXT conversion start
        # get 8 windows
        # train_x = train_x[:, :window_size * n_sample, :]
        # train_y = train_y[:, r_i, :]
        # test_x = test_x[:, :window_size * n_sample, :]
        # test_y = test_y[:, r_i, :]

        # test_x = np.array(random_test_x)
        # train_x = np.array(train_x)
        #
        # train_x = np.squeeze(train_x, axis=(2,))
        # test_x = np.squeeze(test_x, axis=(2,))
        #
        # print('train_x m, s: ', train_x.mean(), train_x.std())
        # print('train_x min, max: ', train_x.min(), train_x.max())
        #
        # print('train_x shape : ', train_x.shape)
        # print('train_y shape : ', train_y.shape)
        # print('test_x shape : ', test_x.shape)
        # print('test_y shape : ', test_y.shape)
        # gt = np.argmax(test_y, axis=1)
        # print('0:', len(gt[gt == 0]))
        # print('1:', len(gt[gt == 1]))
        # print('2:', len(gt[gt == 2]))
        # print('3:', len(gt[gt == 3]))
        # print('gt shape:', gt.shape)

        # data_dir = 'processed_data_2560'
        # data_dir = 'processed_data_5120'
        #
        # if not os.path.exists(data_dir):
        #     os.makedirs(data_dir)
        # train_file_suffix = 'train'
        # test_file_suffix = 'test'
        #
        # file_name_train_x = 'X{}'.format(train_file_suffix)
        # file_name_train_y = 'y{}'.format(train_file_suffix)
        # file_name_test_x = 'X{}'.format(test_file_suffix)
        # file_name_test_y = 'y{}'.format(test_file_suffix)
        #
        # np.savetxt('{}/{}'.format(data_dir, file_name_train_x), train_x, delimiter=',', fmt='%1.8f')
        # np.savetxt('{}/{}'.format(data_dir, file_name_train_y), train_y, delimiter=',', fmt='%1.8f')
        # np.savetxt('{}/{}'.format(data_dir, file_name_test_x), test_x, delimiter=',', fmt='%1.8f')
        # np.savetxt('{}/{}'.format(data_dir, file_name_test_y), test_y, delimiter=',', fmt='%1.8f')
        # print('train_x shape : ', train_x.shape)
        # print('train_y shape : ', train_y.shape)
        # print('test_x shape : ', test_x.shape)
        # print('test_y shape : ', test_y.shape)

        model.fit(
            train_x, train_y,
            batch_size=batch_size,
            epochs=MAX_EPOCHS,
            validation_data=(dev_x, dev_y),
            callbacks=[checkpointer, reduce_lr, stopping])

        # model.load_weights(get_filename_for_saving(save_dir))
        print('model to json: ', get_filename_json_for_saving(save_dir))
        model.save("my_model.h5")
        # import json

        # lets assume `model` is main model
        # model_json = model.to_json()
        # with open(get_filename_json_for_saving(save_dir), "w") as json_file:
        #     json.dump(model_json, json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--experiment", "-e", help="tag with experiment name",
                        default="default")
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    train(args, params)
