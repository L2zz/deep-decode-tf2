"""
module for decoding with autoencoder
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
HIDDEN1 = 256
OUTPUT = INPUT


class AE(Model):
    """
    Class for defining autoencoder
    """

    def __init__(self):
        super(AE, self).__init__()

        regularizer = regularizers.l2(0.005)

        self.input_layer = layers.Input(shape=(INPUT,))
        self.hidden1 = layers.Dense(
            HIDDEN1, activation="elu", kernel_regularizer=regularizer)
        self.output_layer = layers.Dense(OUTPUT)
        self.drop_out = layers.Dropout(DROPOUT_PROB)

        optimizer = tf.optimizers.Adam(LEARNING_RATE)

        self.autoencoder = self.build_model()
        self.autoencoder.compile(loss="mse", optimizer=optimizer)
        self.autoencoder.summary()

    def build_model(self):
        """
        Connect layers and build model
        """
        hidden1 = self.hidden1(self.input_layer)
        drop_out1 = self.drop_out(hidden1)
        output_layer = self.output_layer(drop_out1)

        return Model(self.input_layer, output_layer)

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
        rev_arr = []
        for i in tqdm(range(len(train)), desc="Prepare Train", ncols=80):
            sig_arr.append(train[i].values)
            epc_arr.append(train[i].answer)
            _, reverse = rb.Signal.detect_preamble(train[i].values)
            rev_arr.append(reverse)
        self.autoencoder.fit(np.array(sig_arr), rb.Signal.gen_signal(epc_arr, rev_arr), verbose=1,
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
            pred = self.autoencoder.predict(np.array(sig_arr[fn]))
            for i in tqdm(range(len(pred)), desc=fn, ncols=80):
                pre_idx, reverse = rb.Signal.detect_preamble(pred[i])
                decoded = rb.Signal.detect_data(pred[i][pre_idx:], reverse)
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
    EPOCHS = 100
    PATIENCE = 5
    DROPOUT_PROB = 0.
    VALID_SPLIT = 0.2
    TEST_SPLIT = 0.2
    BATCH_SIZE = 100
    TRAIN_SET_SIZE = BATCH_SIZE

    TRAIN_DATA_DIR = "data_good"
    TEST_DATA_DIR = "data"
    MAX_NUM_SIG = 500

    model = AE()

    train_files = rd.files_from_dir(TRAIN_DATA_DIR)
    test_files = rd.files_from_dir(TEST_DATA_DIR)
    train_gen = rd.read_files_gen(train_files, MAX_NUM_SIG, TRAIN_SET_SIZE)
    test_gen = rd.read_files_gen(test_files, int(MAX_NUM_SIG * TEST_SPLIT),
                                 int(TRAIN_SET_SIZE * TEST_SPLIT), MAX_NUM_SIG)
    num_train_set = MAX_NUM_SIG // TRAIN_SET_SIZE

    test_data = {}
    for train_idx in range(num_train_set):
        print("\n<<[{}/{}] Train Start! >>".format(train_idx + 1, num_train_set))

        train_set = next(train_gen)
        train_data = []
        for fn in sorted(train_set):
            train_data += train_set[fn]

        test_set = next(test_gen)
        for fn in sorted(test_set):
            try:
                test_data[fn] += test_set[fn]
            except Exception as ex:
                test_data[fn] = []
                test_data[fn] += test_set[fn]
        print()

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
