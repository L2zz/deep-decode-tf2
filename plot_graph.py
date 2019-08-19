from read_dir import read_file
import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":
    FILE_NAME = [sys.argv[1]]
    MAX_NUM_SIG = int(sys.argv[2])
    data_arr = read_file(FILE_NAME, MAX_NUM_SIG)

    for fn, data in data_arr.items():
        for signal in data:
            plt.plot(signal.values)
            plt.show()
