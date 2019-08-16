"""
module for decoding with autoencoder
TODO:
    - Make batch from data_set
    - Apply input class with Signal
    - Separate result according to file name
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
from signal_prep import Signal


# Model Parameters
INPUT = 6800
HIDDEN1 = 4096
HIDDEN2 = 2048
HIDDEN3 = HIDDEN1
OUTPUT = INPUT

class AE(Model):
    def __init__(self):
        super(AE, self).__init__()
        self.regularizer = regularizers.l2(0.005)
        self.hidden1 = layers.Dense(
            HIDDEN1, input_shape=(INPUT,), activation="elu", kernel_regularizer=self.regularizer)
        self.hidden2 = layers.Dense(
            HIDDEN2, activation="elu", kernel_regularizer=self.regularizer)
        self.hidden3 = layers.Dense(
            HIDDEN3, activation="elu", kernel_regularizer=self.regularizer)
        self.output_layer = layers.Dense(OUTPUT)
        self.dp = layers.Dropout(DROPOUT_PROB)

    def call(self, x, is_training=False):
        if is_training:
            x = self.dp(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.output_layer(x)

        return x


def train_step(input):

    with tf.GradientTape() as g:
        sig_arr = []
        epc_arr = []
        for signal in input:
            sig_arr.append(signal.values)
            epc_arr.append(signal.epc)
        pred = model(np.array(sig_arr), is_training=True)
        loss = tf.reduce_mean(tf.square(pred - gen_signal(epc_arr)))
    trainable_variables = model.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer = tf.optimizers.Adam(LEARNING_RATE)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss


def test_step(input):

    results = {}
    sig_arr = {}
    epc_arr = {}
    for fn in input:
        results[fn] = [0, 0]
        sig_arr[fn] = []
        epc_arr[fn] = []
        for signal in input[fn]:
            sig_arr[fn].append(signal.values)
            epc_arr[fn].append(signal.epc)
        pred = model(np.array(sig_arr[fn]))
        for i in tqdm(range(len(pred))):
            pre_idx = rb.detect_preamble(pred[i])
            decoded = rb.detect_data(pred[i][pre_idx:])
            if decoded == epc_arr[fn][i]:
                results[fn][0] += 1
            else:
                results[fn][1] += 1

        print("[{}] SUC: {} | FAIL: {} | ACC: {:.2f}%".format(
            fn, results[fn][0], results[fn][1],
            float(results[fn][0]) / (results[fn][0] + results[fn][1]))*100)

    return results


def gen_signal(epc_arr):

    outs = []

    # Bit mask
    type0a = [0.5] * LEN_HALF_BIT + [-0.5] * LEN_HALF_BIT
    type0b = [-0.5] * LEN_HALF_BIT + [0.5] * LEN_HALF_BIT
    type1a = [-0.5] * LEN_BIT
    type1b = [0.5] * LEN_BIT

    # Preamble bit
    preamble = [.5] * LEN_BIT  # 1
    preamble += [-0.5] * LEN_HALF_BIT  # 2
    preamble += [.5] * LEN_HALF_BIT
    preamble += [-0.5] * LEN_BIT  # 3
    preamble += [.5] * LEN_HALF_BIT  # 4
    preamble += [-0.5] * LEN_HALF_BIT
    preamble += [-0.5] * LEN_BIT  # 5
    preamble += [.5] * LEN_BIT  # 6

    padding = (INPUT - (NUM_BIT + NUM_PREAMBLE) * LEN_BIT) // 2

    for epc in epc_arr:
        state = type1b
        out = []
        out += [-0.5] * padding
        out += preamble
        for nbits in range(NUM_BIT):
             # FM0 state transition
            if state == type1b or state == type0b:
                if epc[nbits] == 0:
                    state = type0b
                else:
                    state = type1a
            elif state == type0a or state == type1a:
                if epc[nbits] == 0:
                    state = type0a
                else:
                    state = type1b
            out = out + state

        # dummy bit
        if state == type0a or state == type1a:
            out += [0.5] * LEN_BIT
            out += [-0.5] * (padding - LEN_BIT)
        else:
            out += [-0.5] * padding
        outs.append(np.array(out))

    outs = np.array(outs)

    return outs


if __name__ == "__main__":

    NUM_BIT = 128
    LEN_BIT = 50
    LEN_HALF_BIT = LEN_BIT // 2
    NUM_PREAMBLE = 6
    LEN_PREAMBLE = LEN_BIT * NUM_PREAMBLE

    LEARNING_RATE = 0.0005
    NUM_FOLD = 5
    EPOCHS = 50
    DROPOUT_PROB = 0.2
    PATIENCE = 3

    DATA_DIR = sys.argv[1]

    files = rd.file_from_dir(DATA_DIR)
    data_set = rd.read_file(files)

    # Make train/test data set
    train_data = [[] for i in range(NUM_FOLD)]
    test_data = [{} for i in range(NUM_FOLD)]

    kf = KFold(n_splits=NUM_FOLD, shuffle=True)
    for fn in sorted(data_set):
        for i in range(NUM_FOLD):
            test_data[i][fn] = []
        fold = 0
        for train_idx, test_idx in kf.split(data_set[fn]):
            train_data[fold] += [data_set[fn][i] for i in train_idx]
            test_data[fold][fn] += [data_set[fn][i] for i in test_idx]
            fold += 1

    model = AE()

    for fold in range(NUM_FOLD):
        print("Train Start!")
        patience = 0
        min_loss = 987654321
        for epoch in range(EPOCHS):
            loss = train_step(train_data[fold])
            print("[Epoch {}] LOSS: {:.6f}".format(epoch + 1, loss))
            if loss < 0.03:
                break
            if min_loss < loss:
                patience += 1
                if patience == PATIENCE:
                    break
            else:
                min_loss = loss
                patience = 0

        print("Test Start!")
        results = test_step(test_data[fold])
        suc = 0
        fail = 0
        for fn in results:
            suc += results[fn][0]
            fail += result[fn][1]
        print("[TOTAL] SUC: {} | FAIL: {} | ACC: {:.2f}%".format(
            suc, fail, float(suc/(suc+fail))*100))
        print()
