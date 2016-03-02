import numpy as np

def read_file(file_name) :
    X = []
    y = []
    N = 0
    with open(file_name) as f :
        for line in f :
            data = line.split()
            if N == 0 :
                dim = len(data) - 1
            X.append(np.zeros(dim + 1))
            X[N][0] = 1.0
            for i in xrange(dim) :
                X[N][i + 1] = float(data[i])
            y.append(int(data[dim]))
            N = N + 1
    return np.array(X), np.array(y), N, dim
