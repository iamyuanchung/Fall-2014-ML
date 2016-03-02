import numpy as np
import Reader

if __name__ == '__main__' :

    X_train, y_train = Reader.load('./hw6_train.dat.txt')
    X_test, y_test = Reader.load('./hw6_test.dat.txt')

    T = 300
    u = np.ones(len(X_train)) / len(X_train)
    data_train = np.append(X_train, y_train.reshape(-1, 1), axis = 1)
    for t in xrange(T) :
	# obtain current best decision stump (with lowest Ein)
	min_err = np.inf
	theta = None
	s = None
	feature = None

	data_train = data_train[data_train[ : , 0].argsort()]	# sort with 1st feature
	y_hat = np.zeros(len(data_train))	# y_hat = [+1 +1 +1 ... +1]
	for i in xrange(len(data_train)) :
	    if i > 0 :
		y_hat[i - 1] = -1
	    cur_err = np.sum(u * np.abs((y_hat - data_train[:, 2]) / 2))
	    if cur_err < min_err :
		min_err = cur_err
		if i == 0 :
		    theta = -np.inf
		else :
		    theta = (data_train[i - 1][0] + data_train[i][0]) / 2
		feature = 0
		s = 1
	y_hat[-1] = -1				# y_hat = [-1 -1 -1 ... -1]
	cur_err = np.sum(u * np.abs((y_hat - data_train[:, 2]) / 2))
	if cur_err < min_err :
	    min_err = cur_err
	    theta = np.inf
	    feature = 0
	    s = 1
	for i in xrange(1, len(data_train)) :
	    y_hat[i - 1] = 1
            cur_err = np.sum(u * np.abs((y_hat - data_train[:, 2]) / 2)) 
            if cur_err < min_err :
                min_err = cur_err
                theta = (data_train[i - 1][0] + data_train[i][0]) / 2
                feature = 0
                s = -1

	data_train = data_train[data_train[ : , 1].argsort()]	# sort with 2nd feature
        y_hat = np.zeros(len(data_train))       # y_hat = [+1 +1 +1 ... +1]
        for i in xrange(len(data_train)) :
            if i > 0 :
                y_hat[i - 1] = -1
            cur_err = np.sum(u * np.abs((y_hat - data_train[:, 2]) / 2))
            if cur_err < min_err :
                min_err = cur_err
                if i == 0 :
                    theta = -np.inf
                else :
                    theta = (data_train[i - 1][1] + data_train[i][1]) / 2
                feature = 1
                s = 1
        y_hat[-1] = -1                          # y_hat = [-1 -1 -1 ... -1]
        cur_err = np.sum(u * np.abs((y_hat - data_train[:, 2]) / 2))
        if cur_err < min_err :
            min_err = cur_err
            theta = np.inf
            feature = 1
            s = 1
        for i in xrange(1, len(data_train)) :
            y_hat[i - 1] = 1
            cur_err = np.sum(u * np.abs((y_hat - data_train[:, 2]) / 2))
            if cur_err < min_err :
                min_err = cur_err
                theta = (data_train[i - 1][1] + data_train[i][1]) / 2
                feature = 1
                s = -1

	# update u
	y_hat = np.zeros(len(X_train))
	for i in xrange(len(y_hat)) :
	    y_hat[i] = s * ((X_train[i][feature] > theta) * 2 - 1)

	eps = np.sum(u * np.abs((y_hat - y_train) / 2)) / np.sum(u)
	diamind = np.sqrt((1 - eps) / eps)
