"""
This uses RandomForest from Scikit-learn, with Stratified K-fold
"""
import csv
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold

def accuracy(Y_predict, Y_test):
    equal = 0
    for i in xrange(len(Y_predict)):
        if Y_predict[i] == Y_test[i]:
            equal += 1

    print 'Accuracy = %s' % (float(equal)/len(Y_predict))

data = [ i for i in csv.reader(file('wine.data', 'rb')) ]
random.shuffle(data)

X = np.array([ i[1:] for i in data ])
Y = np.array([ i[0] for i in data ])
K = 5

skf = StratifiedKFold(Y, K)
for train_index_vector, test_index_vector in skf:
    X_train = X[train_index_vector]
    Y_train = Y[train_index_vector]
    X_test = X[test_index_vector]
    Y_test = Y[test_index_vector]
    
    classifier = RandomForestClassifier(n_estimators=10)
    classifier = classifier.fit(X_train, Y_train)
    Y_predict = classifier.predict(X_test)
    
    accuracy(Y_predict, Y_test)
