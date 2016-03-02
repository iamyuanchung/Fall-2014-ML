import numpy as np
import file_reader as fr
from linear_model import PLA

if __name__ == '__main__' :

    X, y, N, dim = fr.read_file('hw1_train_15.txt')

    pla = PLA()

    print '< Question 15 >'
    print 'number of updates : ' + str(pla.fit(X, y))
