# #Standard Libraries
import sys
import os
from os import walk

import json
import threading

import argparse


import numpy as np


import mxnet as mx

##Keras Imports
import keras
from keras import backend as K
from keras.layers import Dense, Dropout, Input, BatchNormalization, Activation

from keras.models import Sequential
from keras.models import save_mxnet_model
from keras.models import load_model
from keras.optimizers import RMSprop, Adam, SGD

from keras.utils import multi_gpu_model


input_shape = (6003,)


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        print('I am here:')
        Matrix = []
        print(type(request_body))
        for line in request_body.splitlines():
            data = line.split()
            target = float(data[0])
            row = np.zeros(input_shape[0], float)
            for i, (idx, value) in enumerate([item.split(':') for item in data[1:]]):
                row[int(idx)] = value
            Matrix.append(np.array(row))
        Matrix = np.array(Matrix)
        print('Matrix Shape', Matrix.shape)
        print('Matrix Type', type(Matrix))
        return mx.io.NDArrayIter(Matrix)
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        pass



class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


def readfile(filepath):
    with open(filepath) as fp:
        lines = fp.readlines()
        Matrix = []
        labels = []
        for line in lines:
            data = line.split()
            target = labels.append(float(data[0]))
            row = np.zeros(input_shape[0], float)
            for i, (idx, value) in enumerate([item.split(':') for item in data[1:]]):
                row[int(idx)] = value
            Matrix.append(np.array(row))
        Matrix = np.array(Matrix)
        return labels, Matrix


def generator(files, batch_size):
    # print('start generator')
    while 1:
        # print('loop generator')
        for file in files:
            try:
                # data = load_svmlight_file(file)
                Y, X = readfile(file)
                recs = X.shape[0]
                batches = int(np.ceil(recs / batch_size))
                for i in range(0, batches):
                    x_batch = X[i * batch_size:min(len(X), i * batch_size + batch_size), ]
                    y_batch = Y[i * batch_size:min(len(X), i * batch_size + batch_size)]
                    yield x_batch, y_batch
            except EOFError:
                print("error" + file)


def parse_args():
    parser = argparse.ArgumentParser()

    # retrieve the hyperparameters we set in notebook (with some defaults)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.01)

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training_channel', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test_channel', type=str, default=os.environ['SM_CHANNEL_TEST'])

    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    parser.add_argument('--batch_size', type=int, default=1024)

    return parser.parse_args()


def train(current_host, hosts, num_cpus, num_gpus, training_dir, val_dir, model_dir, batch_size, epochs, learning_rate):
    print('Parameters: ', num_cpus, num_gpus, training_dir, val_dir, model_dir, batch_size, epochs, learning_rate)

    train_files = [os.path.join(training_dir, file) for file in os.listdir(training_dir)]
    train_files = [x for x in train_files if x.split('/')[-1] != '.ipynb_checkpoints']
    train_gen = generator(train_files, batch_size)
    print('Number of training files: ', len(train_files))

    test_files = [os.path.join(val_dir, file) for file in os.listdir(val_dir)]
    test_files = [x for x in test_files if x.split('/')[-1] != '.ipynb_checkpoints']
    test_gen = generator(test_files, batch_size)
    print('Number of test files: ', len(test_files))

    # shape = load_svmlight_file(train_files[0])[0].shape[1:]
    # shape = (scipy.sparse.load_npz(train_files[0]).shape[1] -1 ,)## Substracting 1 for label columns
    print('shape:', input_shape)

    #     steps = np.array([int(np.ceil(scipy.sparse.load_npz(x).shape[0]/batch_size)) for x in train_files]).sum()
    #     steps_val = np.array([int(np.ceil(scipy.sparse.load_npz(x).shape[0]/batch_size)) for x in test_files]).sum()

    steps = np.array([int(np.ceil(readfile(x)[1].shape[0] / batch_size)) for x in train_files]).sum()
    steps_val = np.array([int(np.ceil(readfile(x)[1].shape[0] / batch_size)) for x in test_files]).sum()
    print('Steps', steps, steps_val)
    print('Number of gpus', num_gpus)

    model = Sequential()
    model.add(Dense(1024, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print(model.summary())

    if num_gpus > 1:
        model = multi_gpu_model(model, gpus=num_gpus)

    filepath = 'best_model.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                                 verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    opt = Adam(lr=learning_rate, rescale_grad=1. / batch_size)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[f1], callbacks=callbacks_list)
    # model.fit_generator(train, epochs=epochs, shuffle=True, class_weight=[0.6,0.4],
    #     validation_data=test, callbacks=[es, mc, lrp], max_queue_size=30, verbose=verbose)

    history = model.fit_generator(train_gen, steps_per_epoch=steps, epochs=epochs,
                                  shuffle=True,
                                  validation_data=test_gen,
                                  validation_steps=steps_val,
                                  callbacks=callbacks_list,
                                  verbose=2)

    print(np.sqrt(history.history['loss']))
    print(np.sqrt(history.history['val_loss']))
    print('model trained: ')
    return model


if __name__ == '__main__':
    args = parse_args()
    num_cpus = int(os.environ['SM_NUM_CPUS'])
    num_gpus = int(os.environ['SM_NUM_GPUS'])
    print('number of cpus', num_cpus)
    print('number of gpus', num_gpus)

    os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    model = train(args.current_host, args.hosts, num_cpus, num_gpus, args.training_channel, args.test_channel,
                  args.model_dir,
                  (args.batch_size), args.epochs, args.learning_rate)

    if args.current_host == args.hosts[0]:
        model.save(os.path.join(args.model_dir, 'model'))
        print('model saved')
        # Get our best model
        # best_model = load_model(os.path.join(args.model_dir, 'model'))
        dependencies = {'f1': f1}
        best_model = load_model('best_model.h5', custom_objects=dependencies)
        print('model loaded:-')
        print( os.path.join(args.model_dir, 'model'))
        #
        # Make one random prediction to initialize for the save
        # sample = np.random.randint(args.vocab_size, size=(args.maxlen,))
        # best_model.predict(np.array([sample]))

        test_files = [os.path.join(args.test_channel, file) for file in os.listdir(args.test_channel)]
        # sample = load_svmlight_file(test_files[0])[0].todense()
        Y, X = readfile(test_files[0])
        print(test_files[0])
        print(X.shape)
        best_model.predict(X)

        print( os.path.join(args.model_dir, 'model'))
        data_names, data_shapes = save_mxnet_model(best_model, os.path.join(args.model_dir, 'model'))

        signature = [{'name': data_names[0], 'shape': [dim for dim in data_desc.shape]} for data_desc in data_shapes]
        with open(os.path.join(args.model_dir, 'model-shapes.json'), 'w') as f:
            json.dump(signature, f)
