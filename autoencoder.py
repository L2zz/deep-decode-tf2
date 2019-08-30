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
        for i in tqdm(range(len(train)), desc="Prepare Train", ncols=80):
            sig_arr.append(train[i].values)
            epc_arr.append(train[i].answer)
        self.autoencoder.fit(np.array(sig_arr), rb.Signal.gen_signal(epc_arr), verbose=1,
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
                pre_idx, _ = rb.Signal.detect_preamble(pred[i])
                decoded = rb.Signal.detect_data(pred[i][pre_idx:])
                if decoded == epc_arr[fn][i]:
                    results[fn][0] += 1
                else:
                    results[fn][1] += 1

            print("[{}] SUC: {} | FAIL: {} | ACC: {:.2f}%\n".format(
                fn, results[fn][0], results[fn][1],
                float(results[fn][0]) * 100 / (results[fn][0] + results[fn][1])))

        return results


if __name__ == "__main__":

    LEARNING_RATE = 0.0005
    EPOCHS = 100
    PATIENCE = 5
    DROPOUT_PROB = 0.
    VALID_SPLIT = 0.2
    TEST_SPLIT = 0.2
    BATCH_SIZE = 100
    TRAIN_SET_SIZE = BATCH_SIZE

    DATA_DIR = sys.argv[1]
    MAX_NUM_SIG = int(sys.argv[2])

    files = rd.files_from_dir(DATA_DIR)

    model = AE()

    data_gen = rd.read_files_gen(files, MAX_NUM_SIG, TRAIN_SET_SIZE)
    num_train_set = MAX_NUM_SIG // TRAIN_SET_SIZE

    test_data = {}
    for train_idx in range(num_train_set):
        print("\n<<[{}/{}] Train Start! >>".format(train_idx+1, num_train_set))
        data_set = next(data_gen)
        train_data = []
        for fn in sorted(data_set):
            tmp_train, tmp_test = train_test_split(
                data_set[fn], test_size=TEST_SPLIT, random_state=0)
            train_data += tmp_train
            try:
                test_data[fn] += tmp_test
            except Exception as ex:
                test_data[fn] = []
                test_data[fn] += tmp_test
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
