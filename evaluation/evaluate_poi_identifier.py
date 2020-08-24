#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

clf.fit(features, labels)
print "Overfit accuracy = ", clf.score(features, labels)

clf.fit(features_train, labels_train)
print "Train-Test split accuracy = ", clf.score(features_test, labels_test)

nb_poi_test = [a for a in labels_test if a != 0.0]
print "number of poi in test set = ", len(nb_poi_test)
print "size of test set = ", len(labels_test)

pred = clf.predict(features_test)
true_positives = 0

fmt = '%-8s%-20s%s'
print(fmt % ('', 'Predicted', 'Actual'))
for i, (prediction, actual) in enumerate(zip(pred, labels_test)):
    print(fmt % (i, prediction, actual))
    if prediction == actual == 1.0:
        true_positives += 1

print "Number of true positives = ", true_positives
print

from sklearn import metrics
print "Precision score = ", metrics.precision_score(pred, labels_test)
print "Recall score = ", metrics.recall_score(pred, labels_test)