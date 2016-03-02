from Classifier import LinearRegression
import numpy as np

N = 1000    # number of points
E = 1000    # number of expetiments

def generator() :
    X = np.random.uniform(-1, 1, (N, 2))
    y = (np.sum(X ** 2, axis = 1) > 0.6) * 2 - 1
    for i in xrange(N) :
        if np.random.random() < 0.1 :
            y[i] = -y[i]
    return X, y

def transformer(X) :
    X_trans = np.zeros((len(X), 5))
    for i in xrange(len(X)) :
        X_trans[i, 0] = X[i, 0]
        X_trans[i, 1] = X[i, 1]
        X_trans[i, 2] = X[i, 0] * X[i, 1]
        X_trans[i, 3] = X[i, 0] ** 2
        X_trans[i, 4] = X[i, 1] ** 2
    return X_trans

if __name__ == '__main__' :

    model = LinearRegression()

    # < Question 13 >
    avg_err_rate = 0
    for e in xrange(E) :
        X, y = generator()
        model.fit(X, y)
        avg_err_rate = avg_err_rate + model.error(X, y)
    print '< Question 13 > average Ein = ' + str(avg_err_rate / E) + '\n'

    # < Question 14 >
    X, y = generator()		# generate a set of training examples, also use it to check the agreement
    X = transformer(X)
    model.fit(X, y)
    y_model = (model.predict(X).reshape(len(X)) > 0) * 2 - 1
    max_same = 0
    w_can = np.array([[-0.05, 0.08, 0.13, 1.5, 1.5], [-0.05, 0.08, 0.13, 1.5, 15], [-0.05, 0.08, 0.13, 15, 1.5], [-1.5, 0.08, 0.13, 0.05, 0.05], [-1.5, 0.08, 0.13, 0.05, 1.5]])
    for i in xrange(5) :
	y_can = (np.sum(w_can[i] * X, axis = 1) - 1 > 0) * 2 - 1
	same = sum(y_model == y_can) / float(N)
	if same > max_same :
	    max_same = same
	    k = i
    print '< Question 14 > ' + str(w_can[k]) + '\n'

    # < Question 15 >
    X_train, y_train = generator()	# generate a set of training examples
    X_train = transformer(X_train)
    model.fit(X_train, y_train)
    avg_err_rate = 0
    for e in xrange(E) :		# generate [E] sets of testing examples
        X_test, y_test = generator()
        X_test = transformer(X_test)
        avg_err_rate = avg_err_rate + model.error(X_test, y_test)
    print '< Question 15 > average Eout = ' + str(avg_err_rate / E) + '\n'
