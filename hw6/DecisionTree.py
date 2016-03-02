import Reader
import copy as cp
import numpy as np

bf_num = 0

class CART :	# only suitable for 2D data ...

    def __init__(self) :

	# left sub-tree
	self.left = None	# [CART]

	# right sub-tree
	self.right = None	# [CART]

	# parameter for decision stump
	self.theta = None	# [float]
	self.feature = None	# [int]

	# only used in leaf node
	self.gt = None		# [int]

    def train(self, X, y) :

	if np.abs(np.sum(y)) == len(y) :
	    # terminate condition : all y[i] the same (all +1 or -1)
	    self.gt = y[0]

	else :
	    # learn branching criteria and split data into 2 parts
	    data = np.append(X, y.reshape(-1, 1), axis = 1)

	    # select the decision stump
	    min_imp = np.inf
            data = data[data[ : , 0].argsort()]		# sort with 1st feature
	    for i in xrange(1, len(data)) :
		cur_imp = i * self.Gini(data[: i, 2]) + (len(data) - i) * self.Gini(data[i :, 2])
		if cur_imp < min_imp :
		    min_imp = cur_imp
		    self.theta = (data[i - 1][0] + data[i][0]) / 2
		    self.feature = 0
		    data_left = cp.deepcopy(data[: i, :])
		    data_right = cp.deepcopy(data[i :, :])

            data = data[data[ : , 1].argsort()]		# sort with 2nd feature
	    for i in xrange(1, len(data)) :
		cur_imp = i * self.Gini(data[: i, 2]) + (len(data) - i) * self.Gini(data[i :, 2])
		if cur_imp < min_imp :
		    min_imp = cur_imp
		    self.theta = (data[i - 1][1] + data[i][1]) / 2
		    self.feature = 1
		    data_left = cp.deepcopy(data[: i, :])
		    data_right = cp.deepcopy(data[i :, :])

	    # build sub-tree and return full-tree
	    self.left = CART()
	    self.left.train(data_left[:, : 2], data_left[:, 2].reshape(-1))

	    self.right = CART()
	    self.right.train(data_right[:, : 2], data_right[:, 2].reshape(-1))

	    global bf_num
	    bf_num += 1

    def Gini(self, y) :

	count = np.zeros(2)	# only binary classification is considered ...

	for i in xrange(len(y)) :

	    count[(y[i] + 1) / 2] += 1

	return 1 - np.sum((count / len(y)) ** 2)

    def predict(self, X) :

	y_hat = np.zeros(len(X))

	for i in xrange(len(X)) :	# for each example, apply tree traversal

	    cur_T = self

	    while cur_T.gt == None :	# not reach leaf node yet

		if cur_T.feature == 0 :

		    if X[i][0] > cur_T.theta :

			cur_T = cur_T.right

		    else :

			cur_T = cur_T.left

		elif cur_T.feature == 1 :

		    if X[i][1] > cur_T.theta :

			cur_T = cur_T.right

		    else :

			cur_T = cur_T.left

		else :

		    print 'You shouldn\'t see this line !'

	    y_hat[i] = cur_T.gt

	return y_hat

    def error(self, X, y) :

	y_hat = self.predict(X)

	return np.sum(np.abs(y_hat - y) / 2) / len(y)

if __name__ == '__main__' :

    X_train, y_train = Reader.load('./hw6_train.dat.txt')
    X_test, y_test = Reader.load('./hw6_test.dat.txt')

    # X_train = np.array([[0.3, 0.5], [0.2, 0.7]])
    # y_train = np.array([-1, 1])

    clf = CART()
    clf.train(X_train, y_train)

    # print clf.theta, clf.feature
    # print clf.left.theta, clf.left.feature, clf.left.gt
    # print clf.right.theta, clf.right.feature, clf.right.gt

    # < Question 16 >
    print 'There are %d branch functions in the tree.\n' % bf_num

    # < Question 17, 18 >
    print '(Ein, Eout) = (%f, %f)\n' % (clf.error(X_train, y_train), clf.error(X_test, y_test))

    '''
    # < Question 19 >
    exp_time = 10
    tree_num = 300
    avg_err = 0.
    for each_exp in xrange(exp_time) :
	# create a random forest
	rf_list = []
	for each_tree in xrange(tree_num) :
	    # bootstrapping
	    exp_ind = np.random.choice(a = len(X_train), size = len(X_train), replace = True)
	    exp_X_train = X_train[exp_ind]
	    exp_y_train = y_train[exp_ind]
	    # create and train a decision tree
	    cur_T = CART()
	    cur_T.train(exp_X_train, exp_y_train)
	    # add to the tree list to form a random forest
	    rf_list.append(cur_T)
	y_hat = np.zeros(len(X_test))
	for each_tree in rf_list :
	    y_hat += each_tree.predict(X_test)
	# prediction of the random forest
	y_hat_bag = (y_hat > 0.) * 2 - 1
	cur_err = np.sum(np.abs((y_hat_bag - y_test) / 2)) / float(len(y_test))
	print 'exp #%d : %f' % (each_exp + 1, cur_err)	
	avg_err += cur_err
    print 'average Eout = %f\n' % (avg_err / exp_time)
    '''

    # < Question 20 >
    tree_num = 300
    avg_err = 0.
    # create a random forest
    rf_list = []
    for each_tree in xrange(tree_num) :
        # bootstrapping
        exp_ind = np.random.choice(a = len(X_train), size = len(X_train), replace = True)
        exp_X_train = X_train[exp_ind]
        exp_y_train = y_train[exp_ind]
        # create and train a decision tree
        cur_T = CART()
        cur_T.train(exp_X_train, exp_y_train)
        # add to the tree list to form a random forest
        rf_list.append(cur_T)
    y_hat = np.zeros(len(X_test))
    for each_tree in rf_list :
	y_hat += each_tree.predict(X_test)
	# (a) print each_tree.error(X_test, y_test)
    y_hat_bag = (y_hat > 0.) * 2 - 1
    print 'aggregation : ' + str(np.sum(np.abs((y_hat_bag - y_test) / 2)) / float(len(y_test)))
