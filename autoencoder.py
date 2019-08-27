"""
module for decoding with autoencoder
TODO:
    - Refactor
    - Pydoc
    - Save model(check point)
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys

from tensorflow.keras import Model, layers, callbacks, regularizers
from sklearn.model_selection import KFold
from tqdm import tqdm

import read_dir as rd
import rule_based as rb


# Model Parameters
INPUT = rb.INPUT
HIDDEN1 = 256
OUTPUT = INPUT


class AE(Model):
    def __init__(self):
        super(AE, self).__init__()

        regularizer = regularizers.l2(0.005)

        self.input_layer = layers.Input(shape=(INPUT,))
        self.hidden1 = layers.Dense(
            HIDDEN1, activation="elu", kernel_regularizer=regularizer)
        self.output_layer = layers.Dense(OUTPUT)
        self.dp = layers.Dropout(DROPOUT_PROB)

        optimizer = tf.optimizers.Adam(LEARNING_RATE)

        self.ae = self.build_model()
        self.ae.compile(loss="mse", optimizer=optimizer)
        self.ae.summary()

    def build_model(self):

        h1 = self.hidden1(self.input_layer)
        d1 = self.dp(h1)
        output_layer = self.output_layer(d1)

        return Model(self.input_layer, output_layer)

    def train_model(self, train):

        early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                 patience=PATIENCE, verbose=1, mode='auto')
        sig_arr = []
        epc_arr = []
        rev_arr = []
        for i in tqdm(range(len(train)), desc="Prepare Train", ncols=80):
            sig_arr.append(train[i].values)
            epc_arr.append(train[i].epc)
        history = self.ae.fit(np.array(sig_arr), rb.gen_signal(epc_arr), verbose=1,
                              batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALID_SPLIT,
                              callbacks=[early_stopping])

    def test_model(self, test):

        results = {}
        sig_arr = {}
        epc_arr = {}
        for fn in test:
            results[fn] = [0, 0]
            sig_arr[fn] = []
            epc_arr[fn] = []
            for signal in test[fn]:
                sig_arr[fn].append(signal.values)
                epc_arr[fn].append(signal.epc)
            pred = self.ae.predict(np.array(sig_arr[fn]))
            for i in tqdm(range(len(pred)), desc=fn, ncols=80):
                pre_idx, _ = rb.detect_preamble(pred[i])
                decoded = rb.detect_data(pred[i][pre_idx:])
                if decoded == epc_arr[fn][i]:
                    results[fn][0] += 1
                else:
                    results[fn][1] += 1

            print("[{}] SUC: {} | FAIL: {} | ACC: {:.2f}%".format(
                fn, results[fn][0], results[fn][1],
                float(results[fn][0]) * 100 / (results[fn][0] + results[fn][1])))

        return results


if __name__ == "__main__":

    LEARNING_RATE = 0.0001
    NUM_FOLD = 5
    EPOCHS = 300
    PATIENCE = 5
    DROPOUT_PROB = 0.
    VALID_SPLIT = 0.2
    BATCH_SIZE = 100

    TRAIN_DATA_DIR = "data_good"
    TEST_DATA_DIR = "data"
    MAX_NUM_SIG = 1000

    train_files = rd.file_from_dir(TRAIN_DATA_DIR)
    test_files = rd.file_from_dir(TEST_DATA_DIR)
    train_data_set = rd.read_file(train_files, MAX_NUM_SIG)
    test_data_set = rd.read_file(test_files, MAX_NUM_SIG)

    # Make train/test data set
    train_data = [[] for i in range(NUM_FOLD)]
    test_data = [{} for i in range(NUM_FOLD)]

    kf = KFold(n_splits=NUM_FOLD, shuffle=True)
    for fn in sorted(train_data_set):
        fold = 0
        for train_idx, test_idx in kf.split(train_data_set[fn]):
            train_data[fold] += [train_data_set[fn][i] for i in train_idx]
            train_data[fold] += [train_data_set[fn][i] for i in test_idx]
            fold += 1

    for fn in sorted(test_data_set):
        for i in range(NUM_FOLD):
            test_data[i][fn] = []
        fold = 0
        for train_idx, test_idx in kf.split(test_data_set[fn]):
            test_data[fold][fn] += [test_data_set[fn][i] for i in train_idx]
            test_data[fold][fn] += [test_data_set[fn][i] for i in test_idx]
            fold += 1

    model = AE()

    for fold in range(NUM_FOLD):
        print("\n[{}] Train Start!".format(fold + 1))
        model.train_model(train_data[fold])
        print("\n[{}] Test Start!".format(fold + 1))
        results = model.test_model(test_data[fold])

        suc = 0
        fail = 0
        for fn in results:
            suc += results[fn][0]
            fail += results[fn][1]
        print("\n[TOTAL] SUC: {} | FAIL: {} | ACC: {:.2f}%".format(
            suc, fail, float(suc * 100 / (suc + fail))))
        print()
