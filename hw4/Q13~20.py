import numpy as np
import Reader
from Regression import LinearRegression

if __name__ == '__main__' :

    X_train, y_train = Reader.simple_read('hw4_train.txt')
    X_test, y_test = Reader.simple_read('hw4_test.txt')

    model = LinearRegression()

    # < Question 13 >
    model.fit(X_train, y_train, 10)
    print '< Question 13 > (E_in, E_out) = (%f, %f)\n' % (model.error(X_train, y_train), model.error(X_test, y_test))

    # create lambda
    reg_candidate = 10 ** np.arange(-10., 3)

    # < Question 14 >
    min_E_in = np.inf
    for reg in reg_candidate :
        model.fit(X_train, y_train, reg)
        E_in = model.error(X_train, y_train)
        if E_in <= min_E_in :
            min_E_in = E_in
            min_reg = reg
    model.fit(X_train, y_train, min_reg)
    print '< Question 14 > (lambda, E_in, E_out) = (%e, %f, %f)\n' % (min_reg, min_E_in, model.error(X_test, y_test))

    # < Question 15 >
    min_E_out = np.inf
    for reg in reg_candidate :
        model.fit(X_train, y_train, reg)
        E_out = model.error(X_test, y_test)
        if E_out <= min_E_out :
            min_E_out = E_out
            min_reg = reg
    model.fit(X_train, y_train, min_reg)
    print '< Question 15 > (lambda, E_in, E_out) = (%e, %f, %f)\n' % (min_reg, model.error(X_train, y_train), min_E_out)


    # split X_train into X_train2 and X_val
    X_val, y_val = X_train[120 : ], y_train[120 : ]
    X_train2, y_train2 = X_train[ : 120], y_train[ : 120]

    # < Question 16 >
    min_E_train = np.inf
    for reg in reg_candidate :
        model.fit(X_train2, y_train2, reg)
        E_train = model.error(X_train2, y_train2)
        if E_train <= min_E_train :
            min_E_train = E_train
            min_reg = reg
    model.fit(X_train2, y_train2, min_reg)
    print '< Question 16 > (lambda, E_train, E_val, E_out) = (%e, %f, %f, %f)\n' % (min_reg, min_E_train, model.error(X_val, y_val), model.error(X_test, y_test))

    # < Question 17 >
    min_E_val = np.inf
    for reg in reg_candidate :
        model.fit(X_train2, y_train2, reg)
        E_val = model.error(X_val, y_val)
        if E_val <= min_E_val :
            min_E_val = E_val
            min_reg = reg
    model.fit(X_train2, y_train2, min_reg)
    print '< Question 17 > (lambda, E_train, E_val, E_out) = (%e, %f, %f, %f)\n' % (min_reg, model.error(X_train2, y_train2), min_E_val, model.error(X_test, y_test))

    # < Question 18 >
    model.fit(X_train, y_train, min_reg)
    print '< Question 18 > (E_in, E_out) = (%f, %f)\n' % (model.error(X_train, y_train), model.error(X_test, y_test))

    # < Question 19 >
    X_train, y_train = np.array(X_train), np.array(y_train)
    m = len(X_train) / 5
    min_E_cv = np.inf
    for reg in reg_candidate :
	E_cv = 0
	for i in xrange(5) :
	    model.fit(np.delete(X_train, np.s_[i * m : (i + 1) * m], axis = 0), np.delete(y_train, np.s_[i * m : (i + 1) * m]), reg)
	    E_cv = E_cv + model.error(X_train[i * m : (i + 1) * m], y_train[i * m : (i + 1) * m])
	E_cv = E_cv / 5
	if E_cv <= min_E_cv :
	    min_E_cv = E_cv
	    min_reg = reg
    print '< Question 19 > (lambda, E_cv) = (%e, %f)\n' % (min_reg, min_E_cv)

    # < Question 20 >
    model.fit(X_train, y_train, min_reg)
    print '< Question 20 > (E_in, E_out) = (%f, %f)' % (model.error(X_train, y_train), model.error(X_test, y_test))
