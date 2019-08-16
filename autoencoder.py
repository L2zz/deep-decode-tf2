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
        pred = pred.numpy()
        for i in tqdm(range(len(pred)), desc=fn, ncols=80):
            pre_idx = detect_preamble(pred[i])
            decoded = detect_data(pred[i][pre_idx:])
            if decoded == epc_arr[fn][i]:
                results[fn][0] += 1
            else:
                results[fn][1] += 1

        print("[{}] SUC: {} | FAIL: {} | ACC: {:.2f}%".format(
            fn, results[fn][0], results[fn][1],
            float(results[fn][0])*100 / (results[fn][0] + results[fn][1])))

    return results


def detect_preamble(in_data):
    # preamble mask
    mask = [1.0] * LEN_BIT  # 1
    mask += [-1.0] * LEN_HALF_BIT  # 2
    mask += [1.0] * LEN_HALF_BIT
    mask += [-1.0] * LEN_BIT  # 3
    mask += [1.0] * LEN_HALF_BIT  # 4
    mask += [-1.0] * LEN_HALF_BIT
    mask += [-1.0] * LEN_BIT  # 5
    mask += [1.0] * LEN_BIT  # 6

    max_idx = 0
    max_score = -987654321
    exp_range = range(40, 100)
    for i in exp_range:
        score = 0.0  # correlation score
        for j in range(len(mask)):
            score += in_data[i + j] * mask[j]
        if max_score < score:
            max_idx = i
            max_score = score

    return max_idx


def detect_data(in_data):

    start = int(LEN_PREAMBLE - LEN_BIT * 0.5)
    new_data = list()

    for s in in_data[start:]:
        new_data.append(s)

    in_data = new_data + [0 for _ in range(LEN_BIT)]

    ret = []
    window = range(-1, 2)

    # FM0 mask
    mask0a = ((-1, ) * LEN_HALF_BIT + (1, ) * LEN_HALF_BIT) * 2
    mask0b = ((1, ) * LEN_HALF_BIT + (-1, ) * LEN_HALF_BIT) * 2
    mask1a = (1, ) * LEN_HALF_BIT + (-1, ) * LEN_HALF_BIT + \
        (-1, ) * LEN_HALF_BIT + (1, ) * LEN_HALF_BIT
    mask1b = (-1, ) * LEN_HALF_BIT + (1, ) * LEN_HALF_BIT + \
        (1, ) * LEN_HALF_BIT + (-1, ) * LEN_HALF_BIT
    data = {mask0a: 0, mask0b: 0, mask1a: 1, mask1b: 1}
    state = 1

    cur_index = 0
    for nbits in range(NUM_BIT):
        max_score = -987654321
        max_tmp_index = cur_index
        for cand in window:
            if state == 1:
                scores = {mask0b: 0, mask1a: 0}
            else:
                scores = {mask0a: 0, mask1b: 0}

            tmp_index = cur_index + cand
            if (tmp_index < 0):
                tmp_index = 0

            i = tmp_index
            j = i + LEN_BIT * 2
            chunk = in_data[i:j]

            for mask in scores:
                for i, sample in enumerate(mask):
                    if i < len(chunk):
                        scores[mask] += chunk[i] * float(sample)
                if max_score < scores[mask]:
                    max_mask = mask
                    max_score = scores[mask]
                    max_tmp_index = tmp_index

        # FM0 state transition
        if state == 1 and data[max_mask] == 1:
            state = 0
        elif state == 0 and data[max_mask] == 1:
            state = 1
        ret.append(data[max_mask])
        cur_index = max_tmp_index + LEN_BIT

    return ret


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
    PATIENCE = 5

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
            suc, fail, float(suc*100/(suc+fail))))
        print()
