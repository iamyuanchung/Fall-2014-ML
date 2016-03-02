import numpy as np

class LinearRegression :

    def __init__(self) :

	self.w = []

    def fit(self, X, y, reg) :

	X = np.append(np.ones((len(X), 1)), np.array(X), axis = 1)
	y = np.array(y).reshape(len(y), 1)

	self.w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + reg * np.eye(len(X[0]))), X.T), y)

    def predict(self, X) :

	X = np.append(np.ones((len(X), 1)), np.array(X), axis = 1)

	return np.dot(X, self.w)

    def error(self, X, y) :

	y_hat = (self.predict(X).reshape(len(X)) > 0) * 2 - 1
	y = np.array(y)

	return 1 - np.sum(np.abs((y_hat + y) / 2)) / float(len(X))
