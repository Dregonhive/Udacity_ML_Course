#!/usr/bin/python

'''

exercised_stock_options :  is the amount of stock options exercised (i.e bought or sold) within a vesting period
restricted_stock : a nontransferable stock that is subject to forfeiture under certain conditions, such as termination of employment or failure to meet either corporate or personal performance benchmarks.
restricted_stock_deferred : A deferred stock is a stock that does not have any rights to the assets of a company undergoing bankruptcy until all common and preferred shareholders are paid.
total_stock_value : total or cummulative amount of all stock values
deferred_income : Deferred income (also known as deferred revenue, unearned revenue, or unearned income) is, in accrual accounting, money received for goods or services which have not yet been delivered. 
	According to the revenue recognition principle, it is recorded as a liability ( because it represents products or services that are owed to a customer) until delivery is made, at which time it is converted into revenue.
deferral_payments : A loan arrangement in which the borrower is allowed to start making payments at some specified time in the future.
'''

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn import metrics


#---------------------------------------------------------------------------------------------------------------------------
def plotFeatures(data_dict, feature1, feature2):
    moment_data_formatted = featureFormat(data_dict, [feature1, feature2, 'poi'])
    for point in moment_data_formatted:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            color = 'red'
        else:
            color = 'green'
        plt.scatter(x, y, color=color)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()

def calculateRatio(target_messages, all_messages):
    if (target_messages in [0.0, 'NaN'] or all_messages in [0.0, 'NaN']):
        return 0.0
    else:
        return target_messages/float(all_messages)


###---------------------------------------------------- Task 1: Select what features you'll use.
# The first feature must be "poi".

features_list = ['poi'] # You will need to use more features
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#Visualizing the dataset
keys = data_dict.keys()
print 'Number of features in total =', len(data_dict[keys[0]].keys())
print 'Dataset size: ', len(keys)
nb_poi = 0
for key, val in data_dict.items():
    if (val['poi'] == 1.0):
        nb_poi = nb_poi + 1
print 'Number of POI\'s in the dataset =',  nb_poi
print 'Percentage of POI\'s in the dataset =', (len(keys) * 1.0)/nb_poi, "%"
print

#Visualizing state of features 
rows = data_dict.values()
features_availability_percentages = []
for i in range(len(data_dict[keys[0]].keys())):
    NaN_count = 0
    for ii in range(len(keys)):
        if rows[ii].values()[i] == 'NaN':
            NaN_count += 1
    
    features_availability_percentages.append([rows[ii].keys()[i], float(len(keys) - NaN_count)/len(keys)])

''''
print "Sorted availability percentages for features (from the the one having the most missing values to the least)"
features_availability_percentages.sort(key=lambda x:x[1])
for i in features_availability_percentages:
    print i[0], "is =", i[1]
'''


#we choose to ignore features that are +40% missing throught the table
threshold = 0.4
all_features = []
for i in features_availability_percentages:
    if i[1] >= threshold:
        all_features.append(i[0])

#Manually removing other features that don't add much to the data
all_features = [e for e in all_features if e not in ('other', 'email_address')]
all_features.remove('poi') #just to append it to the features_list so it would have 'poi' as first feature.
features_list += all_features

print "Features that were taken into account:"
print features_list

###---------------------------------------------------- Task 2: Remove outliers
#Plotting to spot well visible outliers
#plotFeatures(data_dict, 'salary', 'bonus')

#removing outliers
data_dict.pop('TOTAL', 0)
print
#removing rows that have a lot of NaN and0 values, that don't contribute much
useless_rows = []
threshold_rows = 0.85
for name, value in data_dict.items():
    Nb_unavailable_features = 0
    for feature in value.values():
        if (feature == 'NaN' or feature == 0.0):
            Nb_unavailable_features += 1
    percentage = float(Nb_unavailable_features) / float(len(value.items()))
    if percentage > threshold_rows:
        print name, "has a percentage of", percentage, Nb_unavailable_features
        useless_rows.append(name)

for name in useless_rows:
    data_dict.pop(name)

#plotFeatures(data_dict, 'salary', 'bonus')

###---------------------------------------------------- Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#Creating new features: Email ratios like in the previous chapter
for name in my_dataset:
    row = my_dataset[name]
    row["emails_to_poi"] = calculateRatio(row['from_this_person_to_poi'], row['from_messages'])
    row["emails_from_poi"] = calculateRatio(row['from_poi_to_this_person'], row['from_messages'])

features_list += ["emails_to_poi", "emails_from_poi"]
print 
#Select K best features
from sklearn.feature_selection import f_classif, SelectKBest
data = featureFormat(my_dataset, features_list)
k_labels, k_features = targetFeatureSplit(data)
k_best = SelectKBest(f_classif, k = 5)
k_best.fit(k_features, k_labels)
scores = k_best.scores_
pairs = zip(features_list[1:], scores)
pairs.sort(key=lambda x:x[1], reverse = True)
k_best_features = pairs[:10]
features_list = ['poi'] + [x[0] for x in k_best_features]
print features_list
print
#remove other features manually if judged so

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#scaling features
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#scl = MinMaxScaler()
scl = StandardScaler()
features = scl.fit_transform(features)

#splitting 
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, labels, test_size=0.3, random_state=42)

###---------------------------------------------------- Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#------------------------- SVM

def svm_classifier(feature_train, feature_test, target_train, target_test):

    from sklearn import svm
    #parameters = {'kernel':('rbf', 'linear'), 'C':[1, 20, 25, 30, 50], 'max_iter':[1000,50000,100000]}
    #svr = svm.SVC()
    #clf = GridSearchCV(svr, parameters) #It generates a grid of parameter combinations.
    clf = svm.SVC(kernel = 'linear', C = 25, max_iter = 1000)
    clf.fit(feature_train, target_train)
    #print "Best params for SVM:", clf.best_params_

    pred = clf.predict(feature_test)
    print "--------------------- SVM"
    print "Accuracy =", metrics.accuracy_score(pred, target_test)
    print "Precision =", metrics.precision_score(pred, target_test)
    print "Recall =", metrics.recall_score(pred, target_test)
    return clf
#------------------------- Decision Tree
def dt_classifier(feature_train, feature_test, target_train, target_test):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(feature_train, target_train)

    pred = clf.predict(feature_test)
    print "--------------------- Decision Tree"
    print "Accuracy =", metrics.accuracy_score(pred, target_test)
    print "Precision =", metrics.precision_score(pred, target_test)
    print "Recall =", metrics.recall_score(pred, target_test)
    return clf

#------------------------- AdaBoost
def adaboost_classifier(feature_train, feature_test, target_train, target_test):
    from sklearn.ensemble import AdaBoostClassifier
    #parameters = {'n_estimators':[1,2,7,8,9,10]}
    clf = AdaBoostClassifier(n_estimators = 2)
    #clf = GridSearchCV(clf, parameters) #It generates a grid of parameter combinations.
    clf.fit(feature_train, target_train)
    #print "Best params for AdaBoost:", clf.best_params_
    pred = clf.predict(feature_test)

    print "--------------------- AdaBoost"
    print "Accuracy =", metrics.accuracy_score(pred, target_test)
    print "Precision =", metrics.precision_score(pred, target_test)
    print "Recall =", metrics.recall_score(pred, target_test)
    return clf

#------------------------- Naive Bayes
def naive_bayes_classifier(feature_train, feature_test, target_train, target_test):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(feature_train, target_train)

    pred = clf.predict(feature_test)
    print "--------------------- Naive Bayes"
    print "Accuracy =", metrics.accuracy_score(pred, target_test)
    print "Precision =", metrics.precision_score(pred, target_test)
    print "Recall =", metrics.recall_score(pred, target_test)
    return clf

#------------------------- KNN
def knn_classifier(feature_train, feature_test, target_train, target_test):
    from sklearn.neighbors import KNeighborsClassifier
    parameters = {'n_neighbors':[4,5,6,7,8]}
    clf = KNeighborsClassifier(n_neighbors = 4)
    #clf = GridSearchCV(clf, parameters)
    clf = clf.fit(feature_train, target_train)
    #print "Best params for KNN:", clf.best_params_
    pred = clf.predict(feature_test)
    print "--------------------- KNN"
    print "Accuracy =", metrics.accuracy_score(pred, target_test)
    print "Precision =", metrics.precision_score(pred, target_test)
    print "Recall =", metrics.recall_score(pred, target_test)
    return clf

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
##clf = svm_classifier(feature_train, feature_test, target_train, target_test)
##clf = adaboost_classifier(feature_train, feature_test, target_train, target_test)
##clf = naive_bayes_classifier(feature_train, feature_test, target_train, target_test)
clf = dt_classifier(feature_train, feature_test, target_train, target_test)
##clf = knn_classifier(feature_train, feature_test, target_train, target_test)
print "======="
print clf
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)