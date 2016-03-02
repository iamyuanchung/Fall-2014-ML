import numpy as np
import file_reader as fr
from linear_model import PLA

if __name__ == '__main__' :

    X, y, N, dim = fr.read_file('hw1_train_15.txt')

    pla = PLA()
    exp_time = 2000

    print '< Question 17 >'

    total_update = 0
    for i in xrange(exp_time) :
	update = pla.fit(X, y, True, 0.5)
	print 'exp #' + str(i + 1) + ' : ' + str(update)
	total_update = total_update + update

    print 'average number of updates : ' + str(float(total_update) / exp_time)
