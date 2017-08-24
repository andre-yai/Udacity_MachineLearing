#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from __future__ import division

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
del data_dict["TOTAL"]


def compute_fraction_poi_total_messages(poi_messages, total_messages):
    fraction_messages = 0.
    
    if(poi_messages !='NaN' and total_messages != 'NaN'):
        fraction_messages = poi_messages/total_messages
    
    return fraction_messages

def compute_fraction_from_poi_to_person(person_data):

    qtd_messages_total = person_data['to_messages']
    qtd_messages_from_poi = person_data['from_poi_to_this_person']
    fraction_messages_from_poi = compute_fraction_poi_total_messages(qtd_messages_from_poi,qtd_messages_total)
    
    return fraction_messages_from_poi

def compute_fraction_from_person_to_poi(person_data):
    
    qtd_messages_to_total = person_data['from_messages']
    qtd_messages_to_poi = person_data['from_this_person_to_poi']
    fraction_messages_to_poi = compute_fraction_poi_total_messages( qtd_messages_to_poi, qtd_messages_to_total)
    
    return fraction_messages_to_poi


feature_list = data_dict.values()[0].keys()
for person_name in my_dataset:
    person_data = my_dataset[person_name]
    person_data["fraction_messages_from_poi"] = compute_fraction_from_poi_to_person(person_data);
    person_data["fraction_messages_to_poi"] = compute_fraction_from_person_to_poi(person_data);
    print(person_data["fraction_messages_from_poi"],person_data["fraction_messages_to_poi"])
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

new_features = ["fraction_messages_from_poi","fraction_messages_to_poi"]

for feature_name in new_features:
    if feature_name  not in features_list:
        features_list.append(feature_name)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

my_data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(my_data)

# Train and test
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#Function that will help to compute metrics.
from sklearn.metrics import accuracy_score,precision_score,recall_score

def compute_metrics(predict,labels_test):

    accuracy = accuracy_score(predict,labels_test);
    recall = recall_score(predict,labels_test);
    precision = precision_score(predict,labels_test);

    print "Accuracy Score: {}%".format(accuracy) + " Recall Score: {}%".format(recall) + " Precision Score: {}%".format(precision);

# Feature selection - selectionkbest, select percentile,lasso regression
from sklearn.feature_selection import SelectPercentile,SelectKBest,chi2

selector = SelectKBest(k=10)
new_data = selector.fit_transform(features,labels)
scores_KB = selector.scores_

print("Final shape of parameters",new_data.shape)
print("Feature scores:",selector.scores_)
                     
# Provided to give you a starting point. Try a variety of classifiers.                            
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train,labels_train)
predict = clf.predict(features_test)
compute_metrics(predict,labels_test);

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(features_train,labels_train)
predict = clf.predict(features_test)
compute_metrics(predict,labels_test);


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
predict = clf.predict(features_test)
compute_metrics(predict,labels_test);


from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=110)
clf.fit(features_train,labels_train)
predict = clf.predict(features_test)
compute_metrics(predict,labels_test);

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(features_train,labels_train)
predict = clf.predict(features_test)
compute_metrics(predict,labels_test);


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


from sklearn.model_selection import StratifiedKFold,GridSearchCV

def init_cross_validation(train_reduced, labels):
    
    cross_validation = StratifiedKFold(n_splits=3)
    cross_validation.get_n_splits(train_reduced, labels)
    
    return cross_validation;

def generate_grid_search_model(model,parameters,train_reduced,labels):
    
    cross_validation = init_cross_validation(train_reduced, labels);
    
    grid_search = GridSearchCV(model,
                               scoring='accuracy',
                               param_grid = parameters,
                               cv = cross_validation)

    grid_search.fit(train_reduced, labels)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
    return model

def create_AdaBoost(train_reduced,labels,run_grid_search=False):
    
    if run_grid_search:
        
        parameter_grid = {
            'n_estimators': [100,110,120,130]
        }
        clf = AdaBoostClassifier()
        clf = generate_grid_search_model(clf,parameter_grid,train_reduced,labels)
    else: 
        parameters = {
            'n_estimators': 130
        }

        clf = AdaBoostClassifier(**parameters)
        clf.fit(train_reduced, labels)

    return clf
    
clf = create_AdaBoost(features,labels,run_grid_search=True)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)