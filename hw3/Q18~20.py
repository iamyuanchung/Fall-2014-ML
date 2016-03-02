import Reader
import Classifier as CLF

if __name__ == '__main__' :

    X_train, y_train = Reader.simple_read('hw3_train.txt')
    X_test, y_test = Reader.simple_read('hw3_test.txt')

    model = CLF.LogisticRegression()
    
    model.fit(X_train, y_train, 2000, 0.001, False)
    print '< Question 18 > Eout = ' + str(model.error(X_test, y_test)) + '\n'

    model.fit(X_train, y_train, 2000, 0.01, False)
    print '< Question 19 > Eout = ' + str(model.error(X_test, y_test)) + '\n'

    model.fit(X_train, y_train, 2000, 0.001, True)
    print '< Question 20 > Eout = ' + str(model.error(X_test, y_test)) + '\n'
