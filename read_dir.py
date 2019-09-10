"""
module for reading data in target dir
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import rule_based as rb


def files_from_dir(target_dir):
    """
    Get files from target directory.
    @param
        target_dir: target directory name(path)
    @return
        file_list: list of files in target_dir
                   reprensented with <target dir>/<file name>
    """
    file_list = os.listdir(target_dir)
    file_list.sort()

    # represent target directory
    file_list = [target_dir + "/" + file for file in file_list]

    return file_list


def read_file(file, max_num_sig=1000, start_idx=0):
    """
    Read file and return written signals
    @param
        file: target file name
        max_num_sig: maixmum number of signals to get
        start_idx: start index of signal to get
    @return
        Signals: array to save signals
    """
    with open(file, "r") as file_reader:
        signals = []
        ln = 0
        try:
            for _ in range(start_idx):
                file_reader.readline()
                ln += 1
            for _ in tqdm(range(max_num_sig), desc=file, ncols=80):
                line = file_reader.readline()
                ln += 1
                values = line.split(',')
                values.remove('\n')  # Remove newline chararcter at last
                signals.append(rb.Signal(file, values, ln))
        except Exception as ex:
            pass

    return signals


def read_files(file_arr, max_num_sig=1000, start_idx=0):
    """
    Read files and return written signals
    @param
        file_arr: list of target file names
        max_num_sig: maixmum number of signals to get
        start_idx: start index of signal to get
    @return
        Signals: dictionary to save signals
            >> key: file name, value: Signal
    """
    signals = {}
    for file in file_arr:
        tmp = read_file(file, max_num_sig, start_idx)
        if not tmp:
            continue
        signals[file] = tmp

    return signals


def read_files_gen(file_arr, max_num_sig=1000, ret_size=-1, start_idx=0):
    """
    Build generator for read files and yield written signals with ret_size
    @param
        file_arr: list of target file names
        max_num_sig: maixmum number of signals to get
        ret_size: number of siganls to get at once
        start_idx: start index of signal to get
    @yield
        Signals: dictionary to save signals
            >> key: file name, value: Signal
    """
    if ret_size == -1:
        ret_size = max_num_sig

    signals_fn = {}
    ret_num = max_num_sig // ret_size
    for ret_idx in range(ret_num):
        for file_idx in tqdm(range(len(file_arr)), desc="Batch["+str(ret_idx+1)+"]", ncols=80):
            file = file_arr[file_idx]
            with open(file, "r") as file_reader:
                signals = []
                ln = 0
                try:
                    for _ in range(start_idx + ret_idx * ret_size):
                        file_reader.readline()
                        ln += 1
                    for _ in tqdm(range(ret_size), desc=file, ncols=80):
                        line = file_reader.readline()
                        ln += 1
                        values = line.split(',')
                        # Remove newline chararcter at last
                        values.remove('\n')
                        signals.append(rb.Signal(file, values, ln))
                except Exception as ex:
                    pass
            if not signals:
                continue
            signals_fn[file] = signals

        yield signals_fn


def read_files_rand_gen(file_arr, max_num_sig=1000, ret_size=-1, start_idx=0):
    """
    Build generator for read files and yield written shuffled signals with ret_size
    @param
        file_arr: list of target file names
        max_num_sig: maixmum number of signals to get
        ret_size: number of siganls to get at once
        start_idx: start index of signal to get
    @yield
        Signals: dictionary to save signals
            >> key: file name, value: Signal
    """
    if ret_size == -1:
        ret_size = max_num_sig

    signals_fn = {}
    ret_num = max_num_sig // ret_size

    shuffled_idx = np.arange(start_idx + 1, start_idx + max_num_sig + 1, 1)
    np.random.shuffle(shuffled_idx)
    for ret_idx in range(ret_num):
        shuffled_idx_batch = shuffled_idx[ret_idx *
                                          ret_size:(ret_idx + 1) * ret_size]
        shuffled_idx_batch.sort()
        print(shuffled_idx_batch)
        for file_idx in tqdm(range(len(file_arr)), desc="Batch["+str(ret_idx+1)+"]", ncols=80):
            file = file_arr[file_idx]
            with open(file, "r") as file_reader:
                signals = []
                ln = 0
                count = 0
                try:
                    for _ in range(start_idx + ret_idx * ret_size):
                        file_reader.readline()
                        ln += 1
                    for _ in range(start_idx + ret_idx * ret_size,
                                   start_idx + max_num_sig - ret_idx * ret_size):
                        line = file_reader.readline()
                        ln += 1
                        if ln == shuffled_idx_batch[count]:
                            values = line.split(',')
                            # Remove newline chararcter at last
                            values.remove('\n')
                            signals.append(rb.Signal(file, values, ln))
                            count += 1
                        if count == ret_size:
                            break
                except Exception as ex:
                    pass
            if not signals:
                continue
            signals_fn[file] = signals

        yield signals_fn


if __name__ == "__main__":

    OPTION = sys.argv[1]
    TARGET = sys.argv[2]
    MAX_NUM_SIG = int(sys.argv[3])

    if OPTION == 'r':
        FILES = files_from_dir(TARGET)
        DATA_SET = read_files(FILES, MAX_NUM_SIG)
    elif OPTION == 'f':
        DATA_SET = {}
        DATA_SET[TARGET] = read_file(TARGET, MAX_NUM_SIG)

    for key in DATA_SET:
        for i in range(len(DATA_SET[key])):
            print("(" + key + ", " + str(i) + ")")
            plt.plot(DATA_SET[key][i].values)
            plt.show()
