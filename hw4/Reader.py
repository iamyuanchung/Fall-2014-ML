
def simple_read(file_name) :

    X, y = [], []
    with open(file_name) as f :
        for line in f :
            data = line.split()
            X.append(map(float, data[ : -1]))
            y.append(int(data[-1]))

    return X, y
