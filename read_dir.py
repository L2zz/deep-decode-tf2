"""
module for reading data in target dir
"""
import os
import sys
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
        try:
            for _ in range(start_idx):
                file_reader.readline()
            for _ in tqdm(range(max_num_sig), desc=file, ncols=80):
                line = file_reader.readline()
                values = line.split(',')
                values.remove('\n')  # Remove newline chararcter at last
                signals.append(rb.Signal(file, values, len(signals) + 1))
        except Exception as ex:
            print(ex)

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
