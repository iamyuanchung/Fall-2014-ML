import numpy as np

def generate(N) :
    X = np.linspace(-1.0, 1.0, N)
    Y = []
    for i in xrange(N) :
        if X[i] > 0.0 :
            Y.append(1)
        else :
            Y.append(-1)
        if np.random.randint(5) == 0 :  # noise
            Y[i] = -Y[i]
    Y = np.array(Y)
    seg = X[1] - X[0]
    return X, Y, seg

def calc_error(Y1, Y2, theta, s) :
    diff = sum(abs(Y1 - Y2) / 2)
    return float(diff) / len(Y1), (0.5 + 0.3 * s * (np.abs(theta) - 1))

if __name__ == '__main__' :

    N = 20

    exp_time = 5000
    E_in_sum = 0.0
    E_out_sum = 0.0
    for exp in xrange(exp_time) :
        if exp % 100 == 0 :
            print 'exp #' + str(exp)

        X, Y, seg = generate(N)

        Y_sim = np.ones(N)
        E_in, E_out = calc_error(Y_sim, Y, X[N - 1] + seg / 2, -1)

        for i in xrange(N - 1, -1, -1) :
            Y_sim[i] = -1
            Ei_in, Ei_out = calc_error(Y_sim, Y, X[i] - seg / 2, -1)
            if Ei_in < E_in :
                E_in = Ei_in
                E_out = Ei_out

        for i in xrange(N - 1, -1, -1) :
            Y_sim[i] = 1
            Ei_in, Ei_out = calc_error(Y_sim, Y, X[i] - seg / 2, 1)
            if Ei_in < E_in :
                E_in = Ei_in
                E_out = Ei_out

        E_in_sum = E_in_sum + E_in
        E_out_sum = E_out_sum + E_out

    print 'Ein = ' + str(E_in_sum / exp_time) + ' ; Eout = ' + str(E_out_sum / exp_time)
