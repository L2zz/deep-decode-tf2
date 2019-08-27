"""
module for rule based decoding
"""
import numpy as np
import sys

from tqdm import tqdm

import read_dir as rd


# Global variables for determining start/end point of singal
START = 100 + 1
END = -400
INPUT = 7300 - (START-1) + END

NUM_BIT = 128
LEN_BIT = 50
LEN_HALF_BIT = LEN_BIT // 2
NUM_PREAMBLE = 6
LEN_PREAMBLE = LEN_BIT * NUM_PREAMBLE


class Signal:
    values = []  # sample values
    fn = ""  # from filename
    epc = []  # epc data
    data_idx = 0

    def __init__(self, fn, in_data, idx):
        self.values = [float(s) for s in in_data[START:END]]
        self.values = ceiling(self.values, np.percentile(self.values, 2.5),
                              np.percentile(self.values, 97.5))
        self.values = MinMaxScaler(self.values)
        self.values = self.values - np.mean(self.values)

        _, reverse = detect_preamble(self.values)
        if reverse:
            self.values = [-i for i in self.values]

        self.fn = fn
        self.epc = [int(bit) for bit in list(in_data[0])]
        self.data_idx = idx


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


def ceiling(data, min, max):
    for i in range(len(data)):
        if data[i] > max:
            data[i] = max
        elif data[i] < min:
            data[i] = min

    return data

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


def detect_data(in_data):

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

    for i in range(len(epc_arr)):
        out = []

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
        if state == type0a or state == type1a:
            out += [0.5] * LEN_BIT
            out += [-0.5] * (padding - LEN_BIT)
        else:
            out += [-0.5] * padding
        outs.append(np.array(out))

    outs = np.array(outs)

    return outs


if __name__ == "__main__":

    DATA_DIR = sys.argv[1]
    MAX_NUM_SIG = int(sys.argv[2])
    files = rd.file_from_dir(DATA_DIR)
    data_set = rd.read_file(files, MAX_NUM_SIG)

    for fn in data_set:
        suc = 0
        fail = 0
        for i in tqdm(range(len(data_set[fn])), desc=fn, ncols=80):
            pre_idx, _ = detect_preamble(data_set[fn][i].values) # About 80 ~ 90
            decoded = detect_data(data_set[fn][i].values[pre_idx:])
            if decoded == data_set[fn][i].epc:
                suc += 1
            else:
                fail += 1

        print("[{}] SUC: {} | FAIL: {} | ACC: {:.2f}%".format(
              fn, suc, fail, float(suc*100 / (suc + fail))))
