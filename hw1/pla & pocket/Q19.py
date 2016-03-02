import numpy as np
import file_reader as fr
import linear_model as lm

if __name__ == '__main__' :

    X_train, y_train, N_train, dim_train = fr.read_file('hw1_train_18.txt')
    X_test, y_test, N_test, dim_test = fr.read_file('hw1_test_18.txt')

    pocket = lm.Pocket()
    exp_time = 2000

    print '< Question 19 >'

    total_error = 0
    for i in xrange(exp_time) :
	pocket.fit(X_train, y_train, only_for_Q19 = True)
	error = lm.calc_error(X_test, y_test, pocket.w)
	print 'exp #' + str(i + 1) + ' : ' + str(float(error) / N_test)
	total_error = total_error + error

    print 'average error rate : ' + str(float(total_error) / (exp_time * N_test))
