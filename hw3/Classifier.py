import numpy as np

class LinearRegression :    # all in matrix-form (including the return value) !

    def __init__(self) :

        self.w = []

    def fit(self, X, y) :

        X = np.append(np.ones((len(X), 1)), np.array(X), axis = 1)
        y = np.array(y).reshape(len(y), -1)

        self.w = np.dot(np.linalg.pinv(X), y)

    def predict(self, X) :  # based on the given testing data, predict and return real-valued matrix

        X = np.append(np.ones((len(X), 1)), np.array(X), axis = 1)

        return np.dot(X, self.w)

    def error(self, X, y) : # based on the given testing data, predict the result and calculate the error rate (0/1 error measurement)

        y_hat = (self.predict(X).reshape(len(X)) > 0) * 2 - 1
        y = np.array(y)

        return 1 - np.sum(np.abs((y + y_hat) / 2)) / float(len(y))


class LogisticRegression :

    def __init__(self) :

        self.w = []

    def fit(self, X, y, T, lr, SGD) :

        X = np.append(np.ones((len(X), 1)), np.array(X), axis = 1)
        y = np.array(y).reshape(len(y), -1)

        self.w = np.zeros((len(X[0]), 1))

        for t in xrange(T) :

            if SGD == True :
                i = t % len(X)
                grad = (np.sum(self.sigmoid(-y[i] * np.dot(X[i], self.w))) * (-y[i] * X[i])).reshape(len(X[0]), -1)

            else :
                grad = (np.sum(self.sigmoid(-y * np.dot(X, self.w)) * (-y * X), axis = 0)).reshape(len(X[0]), -1) / len(X)

            self.w = self.w - lr * grad

    def sigmoid(self, z) :
        # available for any dimension of data
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X) :

        X = np.append(np.ones((len(X), 1)), np.array(X), axis = 1)

	return np.rint(self.sigmoid(np.dot(X, self.w))) * 2 - 1

    def error(self, X, y) :

	y_hat = self.predict(X).reshape(len(X))
	y = np.array(y)

        return 1 - np.sum(np.abs((y + y_hat) / 2)) / float(len(y))
