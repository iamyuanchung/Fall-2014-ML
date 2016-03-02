import numpy as np

def load(file_name) :
    X, y = [], []
    with open(file_name) as f :
        for line in f :
            data = line.split()
            X.append(map(float, data[ : -1]))
            y.append(int(data[-1]))
    return np.array(X), np.array(y)
