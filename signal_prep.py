"""
module for defining signal and preprocessing
"""
import matplotlib.pyplot as plt
import numpy as np

# Global variables for determining start/end point of singal
START = 100 + 1
END = -400


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
