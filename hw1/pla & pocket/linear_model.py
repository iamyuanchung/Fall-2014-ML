import numpy as np

class PLA :

    def __init__(self) :
        self.w = []
        self.dim = 0

    def fit(self, X, y, random_visit = False, lr = 1.0, to_update = np.inf) :
        N = len(X)
        self.dim = len(X[0]) - 1
        self.w = np.zeros(self.dim + 1)
        visit_order = np.arange(N)
        if random_visit == True :
            np.random.shuffle(visit_order)
        update = 0
        end = -1
        while update < to_update :      # < warning > will not halt if to_update == np.inf and data is not linearly separable
            end = self.find_error(X, y, visit_order, (end + 1) % N)
            if end < 0 :        # no errors
                break
            self.w = self.w + lr * y[visit_order[end]] * X[visit_order[end]]
            update = update + 1
        return update

    def find_error(self, X, y, visit_order, start) :
        N = len(X)
        i = start
        while True :
            score = sum(self.w * X[visit_order[i]])
            if (y[visit_order[i]] * score < 0.0) or (score == 0.0 and y[visit_order[i]]) :
                return i
            i = (i + 1) % N
            if i == start :     # has run through a cycle
                break
        return -1

class Pocket :

    def __init__(self) :
        self.w = []
        self.dim = 0

    def fit(self, X, y, lr = 1.0, to_update = 50, only_for_Q19 = False) :
        N = len(X)
        self.dim = len(X[0]) - 1
        self.w = np.zeros(self.dim + 1)
        w_optimal = self.w
        min_error = calc_error(X, y, w_optimal)
        for i in xrange(to_update) :
            e = self.find_error(X, y)
            if e < 0 :
                break
            self.w = self.w + lr * y[e] * X[e]
            cur_error = calc_error(X, y, self.w)
            if cur_error < min_error :
                w_optimal = self.w
                min_error = cur_error
        if only_for_Q19 == False :
            self.w = w_optimal
        return (i + 1)

    def find_error(self, X, y) :
        N = len(X)
        visit_order = np.arange(N)
        np.random.shuffle(visit_order)
        for i in xrange(N) :
            score = sum(self.w * X[visit_order[i]])
            if (y[visit_order[i]] * score < 0.0) or (score == 0.0 and y[visit_order[i]]) :
                return visit_order[i]
        return -1

def calc_error(X, y, w) :
    N = len(X)
    error_num = 0
    for i in xrange(N) :
        score = sum(w * X[i])
        if (y[i] * score < 0.0) or (y[i] == 1 and score == 0.0) :
            error_num = error_num + 1
    return error_num