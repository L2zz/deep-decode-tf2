import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import read_dir as rd


if __name__ == "__main__":

    TARGET = "data_good"

    files = rd.files_from_dir(TARGET)
    gen = rd.read_files_gen(files, 100, 10, 1000)
    # gen = rd.read_files_rand_gen(files, 100, 10, 1000)

    data = {}
    for _ in range(10):
        batch = next(gen)
        for fn, signals in batch.items():
            print(fn)
            for signal in signals:
                print(signal.signal_idx)
            print()
