"""
module for reading data in target dir
"""
import matplotlib.pyplot as plt
import os
import sys
import csv

from signal_prep import Signal


def file_from_dir(target_dir):
    """
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

def read_file(file_arr, max_num_sig=0):
    """
    @param
        file_arr: list of target file names
    @return
        Signals: dictionary to save signals
            >> key: file name, value: Signal
    """
    signals = {}
    for file in file_arr:
        tmp = []
        with open(file, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for s in reader:
                try:
                    s.remove('')
                except Exception as ex:
                    print(ex)
                tmp.append(Signal(file, s, len(tmp) + 1))
                if max_num_sig == len(tmp):
                    break
        if len(tmp) == 0:
            continue
        signals[file] = tmp
        print(file, " read complete", len(signals[file]))

    return signals

if __name__ == "__main__":

    DATA_DIR = sys.argv[1]
    files = file_from_dir(DATA_DIR)
    data_set = read_file(files)

    for key in data_set:
        for signal in data_set[key]:
            plt.plot(signal.values)
            plt.show()
