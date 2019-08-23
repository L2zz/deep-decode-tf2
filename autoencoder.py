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
from signal_prep import Signal


# Model Parameters
INPUT = 6800
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
            _, reverse = detect_preamble(train[i].values)
            rev_arr.append(reverse)
        history = self.ae.fit(np.array(sig_arr), gen_signal(epc_arr, rev_arr), verbose=1,
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
                pre_idx, reverse = detect_preamble(pred[i])
                decoded = detect_data(pred[i][pre_idx:], reverse)
                if decoded == epc_arr[fn][i]:
                    results[fn][0] += 1
                else:
                    results[fn][1] += 1

            print("[{}] SUC: {} | FAIL: {} | ACC: {:.2f}%".format(
                fn, results[fn][0], results[fn][1],
                float(results[fn][0]) * 100 / (results[fn][0] + results[fn][1])))

        return results


def detect_preamble(in_data):
    # preamble mask
    mask1 = [1.0] * LEN_BIT  # 1
    mask1 += [-1.0] * LEN_HALF_BIT  # 2
    mask1 += [1.0] * LEN_HALF_BIT
    mask1 += [-1.0] * LEN_BIT  # 3
    mask1 += [1.0] * LEN_HALF_BIT  # 4
    mask1 += [-1.0] * LEN_HALF_BIT
    mask1 += [-1.0] * LEN_BIT  # 5
    mask1 += [1.0] * LEN_BIT  # 6

    mask2 = [-i for i in mask1]

    max_idx = 0
    reverse = False
    max_score = -987654321
    exp_range = range(40, 100)
    for i in exp_range:
        score1 = 0.0  # correlation score
        score2 = 0.0
        for j in range(len(mask1)):
            score1 += in_data[i + j] * mask1[j]
            score2 += in_data[i + j] * mask2[j]

        if score1 > score2 and score1 > max_score:
            max_idx = i
            max_score = score1
            reverse = False
        elif score2 > score1 and score2 > max_score:
            max_idx = i
            max_score = score2
            reverse = True

    return max_idx, reverse


def detect_data(in_data, reverse):

    start = int(LEN_PREAMBLE - LEN_BIT * 0.5)
    new_data = list()

    for s in in_data[start:]:
        new_data.append(s)

    in_data = new_data + [0 for _ in range(LEN_BIT)]

    ret = []
    window = range(-3, 4)

    # FM0 mask
    mask0a = ((-1, ) * LEN_HALF_BIT + (1, ) * LEN_HALF_BIT) * 2
    mask0b = ((1, ) * LEN_HALF_BIT + (-1, ) * LEN_HALF_BIT) * 2
    mask1a = (1, ) * LEN_HALF_BIT + (-1, ) * LEN_HALF_BIT + \
        (-1, ) * LEN_HALF_BIT + (1, ) * LEN_HALF_BIT
    mask1b = (-1, ) * LEN_HALF_BIT + (1, ) * LEN_HALF_BIT + \
        (1, ) * LEN_HALF_BIT + (-1, ) * LEN_HALF_BIT
    data = {mask0a: 0, mask0b: 0, mask1a: 1, mask1b: 1}

    if reverse:
        state = 0
    else:
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


def gen_signal(epc_arr, reverse_arr):

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

    for i in range(len(epc_arr)):
        out = []
        if reverse_arr[i]:
            state = type1a
            out += [0.5] * padding
            out += [-i for i in preamble]
        else:
            state = type1b
            out += [-0.5] * padding
            out += preamble

        for nbits in range(NUM_BIT):
             # FM0 state transition
            if state == type1b or state == type0b:
                if epc_arr[i][nbits] == 0:
                    state = type0b
                else:
                    state = type1a
            elif state == type0a or state == type1a:
                if epc_arr[i][nbits] == 0:
                    state = type0a
                else:
                    state = type1b
            out = out + state

        # dummy bit
        if reverse_arr[i]:
            if state == type0b or state == type1b:
                out += [-0.5] * LEN_BIT
                out += [0.5] * (padding - LEN_BIT)
            else:
                out += [0.5] * padding
        else:
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
    EPOCHS = 100
    PATIENCE = 5
    DROPOUT_PROB = 0.
    VALID_SPLIT = 0.2
    BATCH_SIZE = 100

    DATA_DIR = sys.argv[1]
    MAX_NUM_SIG = int(sys.argv[2])

    files = rd.file_from_dir(DATA_DIR)
    data_set = rd.read_file(files, MAX_NUM_SIG)

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
