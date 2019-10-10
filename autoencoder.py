"""
module for decoding with convencoder
TODO:
    - Save model(check point)
"""
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.keras import Model, layers, callbacks, regularizers
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import read_dir as rd
import rule_based as rb


# Model Parameters
INPUT = rb.INPUT
CONV1 = 4
CONV2 = 16
CONV3 = 32
CONV4 = 64
KERNEL_SIZE = 3
POOL_SIZE = 2
OUTPUT = 268


class CE(Model):
    """
    Class for defining convolutional encoder
    """

    def __init__(self):
        super(CE, self).__init__()

        self.convencoder = self.build_model()
        optimizer = tf.optimizers.Adam(LEARNING_RATE)
        self.convencoder.compile(loss="mse", optimizer=optimizer)
        self.convencoder.summary()

    def res_layer(self, x, filters, pooling=False, dropout=0.0):
        temp = x
        temp = layers.Conv1D(filters, 3, padding = "same")(temp)
        temp = layers.BatchNormalization()(temp)
        temp = layers.Activation("elu")(temp)
        temp = layers.Conv1D(filters, 3, padding = "same")(temp)

        x = layers.Add()([temp,layers.Conv1D(filters, 3, padding = "same")(x)])
        if pooling:
            x = layers.MaxPooling1D(2)(x)
        if dropout != 0.0:
            x = layers.Dropout(dropout)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("elu")(x)
        return x

    def build_model(self):
        """
        Connect layers and build model
        """

        input_layer = layers.Input(shape=(INPUT, 1))
        x = layers.Conv1D(CONV1, 3, padding="same")(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("elu")(x)
        x = self.res_layer(x, CONV2, DROPOUT_PROB)
        x = self.res_layer(x, CONV2, DROPOUT_PROB)
        x = self.res_layer(x, CONV2, True, DROPOUT_PROB)
        x = self.res_layer(x, CONV3, DROPOUT_PROB)
        x = self.res_layer(x, CONV3, DROPOUT_PROB)
        x = self.res_layer(x, CONV3, True, DROPOUT_PROB)
        x = self.res_layer(x, CONV4, DROPOUT_PROB)
        x = layers.Flatten()(x)
        x = layers.Dropout(DROPOUT_PROB)(x)
        output_layer = layers.Dense(OUTPUT)(x)

        return Model(input_layer, output_layer)

    def train_model(self, train):
        """
        Method for model trainig
        @param
            train: train data set
        """
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                 patience=PATIENCE, verbose=1, mode='auto')
        sig_arr = []
        epc_arr = []
        for fn in train:
            for signal in train[fn]:
                sig_arr.append(signal.values)
                epc_arr.append(signal.answer)
        sig_arr = np.array(sig_arr).reshape(len(sig_arr), INPUT, 1)
        epc_arr = rb.Signal.gen_halfbit_signal(epc_arr).reshape(len(epc_arr), OUTPUT, 1)
        self.convencoder.fit(sig_arr, epc_arr, verbose=1,
                             batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALID_SPLIT,
                             callbacks=[early_stopping])

    def test_model(self, test):
        """
        Method for model test
        @param
            test: test data set
        """
        results = {}
        sig_arr = {}
        epc_arr = {}
        for fn in test:
            results[fn] = [0, 0]
            sig_arr[fn] = []
            epc_arr[fn] = []
            for signal in test[fn]:
                sig_arr[fn].append(signal.values)
                epc_arr[fn].append(signal.answer)
            sig_arr[fn] = np.array(sig_arr[fn]).reshape(len(sig_arr[fn]), INPUT, 1)
            pred = self.convencoder.predict(sig_arr[fn])
            for i in tqdm(range(len(pred)), desc=fn, ncols=80):
                decoded = rb.Signal.detect_halfbit_data(pred[i])
                if decoded == epc_arr[fn][i]:
                    results[fn][0] += 1
                else:
                    results[fn][1] += 1

            print("[{}] SUC: {} | FAIL: {} | ACC: {:.2f}%\n".format(
                fn, results[fn][0], results[fn][1],
                float(results[fn][0]) * 100 / (results[fn][0] + results[fn][1])))

        return results


if __name__ == "__main__":

    NUM_BIT = 128
    LEN_BIT = 50
    LEN_HALF_BIT = LEN_BIT // 2
    NUM_PREAMBLE = 6
    LEN_PREAMBLE = LEN_BIT * NUM_PREAMBLE

    LEARNING_RATE = 0.0001
    EPOCHS = 300
    PATIENCE = 5
    DROPOUT_PROB = 0.
    VALID_SPLIT = 0.2
    TEST_SPLIT = 0.2
    BATCH_SIZE = 100

    TRAIN_DATA_DIR = "data_good"
    TEST_DATA_DIR = "data"
    MAX_NUM_SIG = 1000

    model = CE()

    train_files = rd.files_from_dir(TRAIN_DATA_DIR)
    test_files = rd.files_from_dir(TEST_DATA_DIR)
    train_data = rd.read_files(train_files, MAX_NUM_SIG)
    test_data = rd.read_files(test_files, int(
        MAX_NUM_SIG * TEST_SPLIT), MAX_NUM_SIG)

    print("\n<< Train Start! >>")
    model.train_model(train_data)

    print("\n<< Test Start! >>")
    results = model.test_model(test_data)

    suc = 0
    fail = 0
    for fn in results:
        suc += results[fn][0]
        fail += results[fn][1]
    print("\n[TOTAL] SUC: {} | FAIL: {} | ACC: {:.2f}%".format(
        suc, fail, float(suc * 100 / (suc + fail))))
    print()
