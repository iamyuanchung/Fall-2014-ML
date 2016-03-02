import numpy as np

def load(file_name) :
    # X for features, y for labels
    X, y = [], []
    with open(file_name) as f :
	for line in f :
	    data = line.split()
	    y.append(float(data[0]))
	    X.append(map(float, data[1 : ]))
    return np.array(X), np.array(y) 
