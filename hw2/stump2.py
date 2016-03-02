from operator import itemgetter
import numpy as np

def read_file(file_name) :
    Data, N = [], 0
    with open(file_name) as f :
        for line in f :
            data = line.split()
            Data.append([])
            for i in xrange(len(data) - 1) :
                Data[N].append(float(data[i]))
            Data[N].append(int(data[-1]))
            N = N + 1
    return Data, N, len(Data[0]) - 1

def calc_error(Y1, Y2) :
    return sum(abs(Y1 - Y2) / 2) / len(Y1)

if __name__ == '__main__' :

    # < Question 19 >
    Data, N, dim = read_file('hw2_train.txt')

    E_in_all = np.inf
    for d in xrange(dim) :

        Data.sort(key = itemgetter(d))

        X = np.array(Data)[ : , d]
        Y = np.array(Data)[ : , -1]
        Y_sim = np.ones(N)

        E_in = calc_error(Y_sim, Y)
        s, theta = 1, -np.inf

        for i in xrange(N - 1) :
            Y_sim[i] = -1
            Ei_in = calc_error(Y_sim, Y)
            if Ei_in < E_in :
                E_in = Ei_in
                seg = X[i + 1] - X[i]
                s, theta = 1, X[i] + seg / 2

        Y_sim[-1] = -1
        Ei_in = calc_error(Y_sim, Y)
        if Ei_in < E_in :
            E_in = Ei_in
            s, theta = 1, np.inf

        for i in xrange(N - 1) :
            Y_sim[i] = 1
            Ei_in = calc_error(Y_sim, Y)
            if Ei_in < E_in :
                E_in = Ei_in
                seg = X[i +1] - X[i]
                s, theta = -1, X[i] + seg / 2

        if E_in < E_in_all :
            E_in_all = E_in
            s_all, theta_all, d_all = s, theta, d + 1

    print 'Ein = ' + str(E_in_all)
    print 's, theta, d = ' + str(s_all) + ', ' + str(theta_all) + ', ' + str(d_all)

    # < Question 20 >
    Data, N, dim = read_file('hw2_test.txt')

    Y = np.array(Data)[ : , -1]
    Y_sim = []
    for i in xrange(N) :
        pre = s_all * (Data[i][d_all - 1] - theta_all)
        if pre > 0 :
            Y_sim.append(1)
        else :
            Y_sim.append(-1)
    E_out = calc_error(np.array(Y_sim), Y)
    print 'Eout = ' + str(E_out)
