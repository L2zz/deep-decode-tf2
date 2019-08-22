"""
module for rule based decoding
"""
import numpy as np
import sys

from tqdm import tqdm

import read_dir as rd
from signal_prep import Signal


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
    window = range(-1, 2)

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


if __name__ == "__main__":

    NUM_BIT = 128
    LEN_BIT = 50
    LEN_HALF_BIT = LEN_BIT // 2
    NUM_PREAMBLE = 6
    LEN_PREAMBLE = LEN_BIT * NUM_PREAMBLE

    DATA_DIR = sys.argv[1]
    MAX_NUM_SIG = int(sys.argv[2])
    files = rd.file_from_dir(DATA_DIR)
    data_set = rd.read_file(files, MAX_NUM_SIG)

    for fn in data_set:
        suc = 0
        fail = 0
        for i in tqdm(range(len(data_set[fn])), desc=fn, ncols=80):
            pre_idx, reverse = detect_preamble(data_set[fn][i].values) # About 80 ~ 90
            decoded = detect_data(data_set[fn][i].values[pre_idx:], reverse)
            if decoded == data_set[fn][i].epc:
                suc += 1
            else:
                fail += 1

        print("[{}] SUC: {} | FAIL: {} | ACC: {:.2f}%".format(
              fn, suc, fail, float(suc*100 / (suc + fail))))
