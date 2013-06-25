"""
A blend of RandomForest, ExtraTress and GradientBoostingClassifier. Accuracy is normally at 97% or 100%!
"""
import csv
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter

def accuracy(Y_predict, Y_test):
    equal = 0
    for i in xrange(len(Y_predict)):
        if Y_predict[i] == Y_test[i]:
            equal += 1

    print 'Accuracy = %s' % (float(equal)/len(Y_predict))

data = [ i for i in csv.reader(file('wine.data', 'rb')) ]
random.shuffle(data)

X = [ i[1:] for i in data ]
Y = [ i[0] for i in data ]

train_cutoff = len(data) * 3/4

X_train = X[:train_cutoff]
Y_train = Y[:train_cutoff]
X_test = X[train_cutoff:]
Y_test = Y[train_cutoff:]

classifiers = [
    RandomForestClassifier(n_estimators=10, criterion='gini'),
    RandomForestClassifier(n_estimators=10, criterion='entropy'),
    ExtraTreesClassifier(n_estimators=10, criterion='gini'),
    ExtraTreesClassifier(n_estimators=10, criterion='entropy'),
    GradientBoostingClassifier(n_estimators=10),
]

Y_predict = [ [] for i in xrange(len(classifiers)) ]

for i, classifier in enumerate(classifiers):
    classifier.fit(X_train, Y_train)
    Y_predict[i] = classifier.predict(X_test)

Y_mean = []
for i in xrange(len(Y_predict[0])):
    counter = Counter([Y_predict[classifier_id][i] for classifier_id in xrange(len(classifiers))])
    vote = counter.most_common(1)[0][0] # if a tie occurs, the smaller predicted number wins
    Y_mean.append(vote)
    
    # Uncomment this code to see the voting and the tie-breaking here
    # for classifier_id in xrange(len(classifiers)):
    #     print '%s %s %s %s %s => %s' % (Y_predict[0][i], Y_predict[1][i], Y_predict[2][i], Y_predict[3][i], Y_predict[4][i], vote)
    
accuracy(Y_mean, Y_test)