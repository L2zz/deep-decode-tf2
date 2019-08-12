"""
module for rule based decoding
"""
import numpy as np
import sys

import read_dir as rd
from signal_prep import Signal


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


def detect_data(in_data, answer):

    start = int(LEN_PREAMBLE - LEN_BIT * 0.5)
    new_data = list()

    for s in in_data[start:]:
        new_data.append(s)

    in_data = new_data + [0 for _ in range(LEN_BIT)]

    ret = []
    window = range(-2, 3)

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

NUM_BIT = 128
LEN_BIT = 50

LEN_HALF_BIT = LEN_BIT // 2

NUM_PREAMBLE = 6
LEN_PREAMBLE = LEN_BIT * NUM_PREAMBLE

if __name__ == "__main__":

    DATA_DIR = sys.argv[1]
    files = rd.file_from_dir(DATA_DIR)
    data_set = rd.read_file(files)

    for fn in data_set:
        suc = 0
        fail = 0
        for signal in data_set[fn]:
            pre_idx = detect_preamble(signal.values)
            decoded = detect_data(signal.values[pre_idx:], signal.epc)
            if decoded == signal.epc:
                suc += 1
            else:
                fail += 1

        print("%s %3d %3d ==> %.2f" %
              (fn, suc, fail, float(suc) / (suc + fail)))
