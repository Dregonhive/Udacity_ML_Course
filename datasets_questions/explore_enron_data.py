#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
print len(enron_data)
print enron_data.keys()[0]

print len(enron_data[enron_data.keys()[0]])

count_salary = 0
count_payments = 0
for i in range(len(enron_data)):
    ##print enron_data.keys()[i], " with: ", enron_data[enron_data.keys()[i]]["total_payments"]
    if enron_data[enron_data.keys()[i]]['salary'] != 'NaN':
        count_salary += 1
    if enron_data[enron_data.keys()[i]]['total_payments'] == 'NaN' and enron_data[enron_data.keys()[i]]['poi'] == 'true':
        count_payments += 1

print count_salary
print "result = ", count_payments
print "Total stock of James Pentice = ", enron_data['PRENTICE JAMES']['total_stock_value']
print "Total emails to POI of Wesley Colwell = ", enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print "Value of exerciced stock options of Jeffrey K Skilling = ", enron_data['SKILLING JEFFREY K']['exercised_stock_options']

