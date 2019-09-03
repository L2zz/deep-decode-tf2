"""
module for rule based decoding
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import read_dir as rd


# Global variables for determining start/end point of singal
START = 100 + 1
END = -400
INPUT = 7300 - (START - 1) + END

# Global variables for signal info
NUM_BIT = 128
LEN_BIT = 50
LEN_HALF_BIT = LEN_BIT // 2
NUM_PREAMBLE = 6
LEN_PREAMBLE = LEN_BIT * NUM_PREAMBLE


class Signal:
    """
    @attr
        values: sample values
        file_name: file name
        answer: answer of signal
        signal_idx: index of signal in file
    """
    values = []
    file_name = ""
    answer = []
    signal_idx = 0

    def __init__(self, file_name, in_data, idx):
        self.values = [float(s) for s in in_data[START:END]]
        self.values = z_scores(self.values)
        self.values = min_max_scaler(self.values)
        self.values = self.values - np.mean(self.values)

        _, reverse = Signal.detect_preamble(self.values)
        if reverse:
            self.values = [-i for i in self.values]

        self.file_name = file_name
        self.answer = [int(bit) for bit in list(in_data[0])]
        self.signal_idx = idx

    @classmethod
    def detect_preamble(cls, sample_val):
        """
        Detect preamble index
        @param
            sample_val: sample values of signal
        @return
            max_idx: preamble index which gets max correlation score
            reverse: whether signal is reversed or not
        """
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
        reverse = False
        max_score = -987654321
        exp_range = range(40, 100)
        for exp_idx in exp_range:
            score = 0.0
            for mask_idx, mask_val in enumerate(mask):
                score += sample_val[exp_idx + mask_idx] * mask_val

            if abs(score) > max_score:
                if score > 0:
                    reverse = False
                else:
                    reverse = True
                max_idx = exp_idx
                max_score = abs(score)

        return max_idx, reverse

    @classmethod
    def detect_data(cls, sig_val):
        """
        Decoding signal
        @param
            sig_val: valid sample values(preamble + data) of signal
            reverse: whether signal is reversed or not
        @return
            decoded: result of decoding
        """
        start = int(LEN_PREAMBLE - LEN_BIT * 0.5)
        signal = list(sig_val[start:]) + [0 for _ in range(LEN_BIT)]

        decoded = []
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
        for _ in range(NUM_BIT):
            max_score = -987654321
            max_tmp_index = cur_index
            for cand in window:
                if state == 1:
                    scores = {mask0b: 0, mask1a: 0}
                else:
                    scores = {mask0a: 0, mask1b: 0}

                tmp_index = cur_index + cand
                if tmp_index < 0:
                    tmp_index = 0

                chunk_start = tmp_index
                chunk_end = chunk_start + LEN_BIT * 2
                chunk = signal[chunk_start:chunk_end]

                for mask in scores:
                    for mask_idx, sample in enumerate(mask):
                        if mask_idx < len(chunk):
                            scores[mask] += chunk[mask_idx] * float(sample)
                    if max_score < scores[mask]:
                        max_mask = mask
                        max_score = scores[mask]
                        max_tmp_index = tmp_index

            # FM0 state transition
            if state == 1 and data[max_mask] == 1:
                state = 0
            elif state == 0 and data[max_mask] == 1:
                state = 1
            decoded.append(data[max_mask])
            cur_index = max_tmp_index + LEN_BIT

        return decoded

    @classmethod
    def gen_signal(cls, ans_arr):
        """
        Generating theoretical signals using answers
        @param
            ans_arr: array of answers of decoding
            reverse_arr: array of info whether signal is reversed or not
        @return
            outs: array of generated signals
        """
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

        for ans_idx, ans in enumerate(ans_arr):
            out = []
            state = type1b
            out += [-0.5] * padding
            out += preamble
            for nbits in range(NUM_BIT):
                 # FM0 state transition
                if state in (type0b, type1b):
                    if ans[nbits] == 0:
                        state = type0b
                    else:
                        state = type1a
                elif state in (type0a, type1a):
                    if ans[nbits] == 0:
                        state = type0a
                    else:
                        state = type1b
                out = out + state

            # dummy bit
            if state in (type0a, type1a):
                out += [0.5] * LEN_BIT
                out += [-0.5] * (padding - LEN_BIT)
            else:
                out += [-0.5] * padding
            outs.append(np.array(out))

        outs = np.array(outs)

        return outs


def min_max_scaler(data):
    """
    Process min-max scaling
    @param
        data: target array of values to scaling
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


def ceiling(data, min_val, max_val):
    """
    @param
        data: target array of values to ceiling
        min_val: min value of ceiling
        max_val: max value of ceiling
    @return
        data: process data
    """

    for idx, val in enumerate(data):
        if val > max_val:
            data[idx] = max_val
        elif val < min_val:
            data[idx] = min_val

    return data


def tukey_fences(data):
    """
    Outlier detection using tukey fences
    @param
        data: target array of values to ceiling
    @return
        data: process data
    """
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + (iqr * 1.5)
    lower_bound = q1 - (iqr * 1.5)
    for idx, val in enumerate(data):
        if val > upper_bound:
            data[idx] = upper_bound
        elif val < lower_bound:
            data[idx] = lower_bound

    return data


def z_scores(data):
    """
    Outlier detection using z scores (threshold default: 3)
    @param
        data: target array of values to ceiling
    @return
        data: process data
    """
    threshold = 2.5
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(i - mean) / std for i in data]
    for z_idx, z_score in enumerate(z_scores):
        if z_score > threshold:
            data[z_idx] = mean + threshold * std
        elif z_score < -threshold:
            data[z_idx] = mean - threshold * std

    return data


def modified_z_scores(data):
    """
    Outlier detection using modified z scores (threshold default: 3.5)
    @param
        data: target array of values to ceiling
    @return
        data: process data
    """
    threshold = 3.5
    median = np.median(data)
    med_abs_dev = np.median([np.abs(sample - median) for sample in data])
    mz_scores = [0.6745 * (sample - median) / med_abs_dev for sample in data]
    for mz_idx, mz_score in enumerate(mz_scores):
        if mz_score > threshold:
            data[mz_idx] = median + threshold * med_abs_dev / 0.6745
        elif mz_score < -threshold:
            data[mz_idx] = median - threshold * med_abs_dev / 0.6745

    return data


if __name__ == "__main__":

    DATA_DIR = sys.argv[1]
    MAX_NUM_SIG = int(sys.argv[2])
    files = rd.files_from_dir(DATA_DIR)
    data_set = rd.read_files(files, MAX_NUM_SIG)
    print()

    for fn in data_set:
        suc = 0
        fail = 0
        for i in tqdm(range(len(data_set[fn])), desc=fn, ncols=80):
            pre_idx, _ = Signal.detect_preamble(data_set[fn][i].values)
            decoded = Signal.detect_data(data_set[fn][i].values[pre_idx:])
            if decoded == data_set[fn][i].answer:
                suc += 1
            else:
                fail += 1

        print("[{}] SUC: {} | FAIL: {} | ACC: {:.2f}%\n".format(
            fn, suc, fail, float(suc * 100 / (suc + fail))))
