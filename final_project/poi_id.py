#!/usr/bin/python

from __future__ import division
import sys
import pickle
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectPercentile,SelectKBest,chi2
import numpy as np


sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import tester

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Load the dictionary containing the dataset
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

## Transforming data dictionary to pandas data Frame
eron_data = pd.DataFrame.from_dict(data_dict, orient = 'index')


### Treating Missing Data 
def PercentageMissin(Dataset):
    """this function will return the percentage of missing values in a dataset """
    if isinstance(Dataset,pd.DataFrame):
        adict={} #a dictionary conatin keys columns names and values percentage of missin value in the columns
        for feature in Dataset.columns:
            count_nan_feature = 0;
            for feature_row in Dataset[feature]:
                if(feature_row == 'NaN'):
                    count_nan_feature += 1  
            adict[feature]=(count_nan_feature*100)/len(Dataset[feature])
        return pd.DataFrame(adict,index=['% of missing'],columns=adict.keys())
    else:
        raise TypeError("can only be used with panda dataframe")

print("Ranking of Missing Data")
print(PercentageMissin(eron_data).mean().sort_values(ascending=False))


payment_fields =['poi','salary','bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments',
            'loan_advances', 'other','expenses', 'director_fees','total_payments']
stock_fields =['poi','exercised_stock_options','restricted_stock','restricted_stock_deferred','total_stock_value']
email_fields =['to_messages','from_messages','from_poi_to_this_person','from_this_person_to_poi','shared_receipt_with_poi']

## Filling missing Data

eron_data.loc[:,payment_fields] = eron_data.loc[:,payment_fields].replace('NaN', 0)
eron_data.loc[:,stock_fields] = eron_data.loc[:,stock_fields].replace('NaN',0) 

eron_poi = eron_data[eron_data.poi == 1]
eron_non_poi = eron_data[eron_data.poi == 0]

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy = 'mean', axis=0)

eron_poi.loc[:, email_fields] = imp.fit_transform(eron_poi.loc[:,email_fields]);
eron_non_poi.loc[:, email_fields] = imp.fit_transform(eron_non_poi.loc[:,email_fields]);

eron_data = eron_poi.append(eron_non_poi)

### Task 2: Treat wrong fields

eron_data[eron_data[payment_fields[1:-1]].sum(axis='columns') != eron_data['total_payments']][payment_fields + ['total_payments']]
eron_data[eron_data[stock_fields[1:-1]].sum(axis='columns') != eron_data['total_stock_value']][stock_fields+['total_stock_value']]

# Treating Belfer Robert
eron_data.loc['BELFER ROBERT','deffered_income'] = -102500
eron_data.loc['BELFER ROBERT','defferral_payments'] = 0
eron_data.loc['BELFER ROBERT','expenses'] = 3285
eron_data.loc['BELFER ROBERT','director_fees'] = 102500
eron_data.loc['BELFER ROBERT','total_payments'] = 3285

eron_data.loc['BELFER ROBERT','exercised_stock_options'] = 0
eron_data.loc['BELFER ROBERT','restricted_stock'] = 44093
eron_data.loc['BELFER ROBERT','restricted_stock_deferred'] = -44093
eron_data.loc['BELFER ROBERT','total_stock_value'] = 0

eron_data.loc['BELFER ROBERT'][payment_fields+stock_fields]

# Treating BHATNAGAR SANJAY
eron_data.loc['BHATNAGAR SANJAY','other'] = 0
eron_data.loc['BHATNAGAR SANJAY','expenses'] = 137864
eron_data.loc['BHATNAGAR SANJAY','director_fees'] = 0
eron_data.loc['BHATNAGAR SANJAY','total_payments'] = 137864

eron_data.loc['BHATNAGAR SANJAY','exercised_stock_options'] = 15456290
eron_data.loc['BHATNAGAR SANJAY','restricted_stock'] = 2604490
eron_data.loc['BHATNAGAR SANJAY','restricted_stock_deferred'] = -2604490
eron_data.loc['BHATNAGAR SANJAY','total_stock_value'] = 15456290

eron_data.loc['BHATNAGAR SANJAY'][payment_fields+stock_fields]


### Task 3: Remove outliers.

# According to the pdf and the data_dict we have some aggregate row. They are "Total" and "THE TRAVEL AGENCY IN THE PARK". So lets remove them.
eron_data = eron_data.drop(["TOTAL","THE TRAVEL AGENCY IN THE PARK"])

# Lets see those with rows with more outliers. To do so we will see the rows that stays less than 25% of the data
# and those with greather than 75%.
outliers = eron_data.quantile(.5) + 1.5 * (eron_data.quantile(.75)-eron_data.quantile(.25))
outlier_pd = pd.DataFrame((eron_data[1:] > outliers[1:]).sum(axis = 1), columns = ['# of outliers']).    sort_values('# of outliers',  ascending = [0]).head(7)
outlier_pd

# We will remove those with more outliers that are not poi.
eron_data_clean = eron_data.drop(['FREVERT MARK A','WHALLEY LAWRENCE G','LAVORATO JOHN J','KEAN STEVEN J'])

# Reamove NaN Fields
nan_pd =pd.DataFrame((eron_data == 0).astype(int).sum(axis=1), columns = ['# of NaN']).\
sort_values('# of NaN',  ascending = [0]).head(7)
nan_pd

print(data_dict['LOCKHART EUGENE E'])

eron_data = eron_data.drop(['LOCKHART EUGENE E'])
del data_dict['LOCKHART EUGENE E']

### Task 4: Selecting features that I will use

## With some analysis I will remove from my analysis the fields.
## advances,director_fees, restricted_stock_deferred and email_address

features_list = [
    'poi', 'to_messages','from_messages', 'from_poi_to_this_person','from_this_person_to_poi', 
    'salary','deferral_payments', 'other','total_payments','bonus', 
    'total_stock_value', 'shared_receipt_with_poi', 'long_term_incentive',
    'exercised_stock_options','deferred_income', 'expenses', 'restricted_stock']

# In the featureFormat it will treat the missing values giving them a 0  value
eron_data = eron_data[features_list]
data = eron_data

## Auxiliar Functions

def convert_dataframe_into_dataset(df):
    """
        Convert Pandas DataFarme to Dataset.
    """
    scaled_df = df.copy()
    scaled_df.iloc[:,1:] = scale(scaled_df.iloc[:,1:])
    my_dataset = scaled_df.to_dict(orient='index')

    return my_dataset

def test_model_tester(clf,my_dataset,featurs_list):
    """
        This function will test our model into the tester file
    """
    tester.dump_classifier_and_data(clf, my_dataset, features_list)
    return tester.main()

def divide_dataset_into_features_labels(my_dataset,features_list):
    """
        Divide Dataset into features and labels
    """
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    return [labels, features]

class DecisionTree:
    
    def __init__(self,features_train, features_test, labels_train):
        """
            Generate Model
        """
        self.clf = DecisionTreeClassifier()
        self.fit(features_train, labels_train)
        self.predict(features_test)
        
    def fit(self,features,labels):
        """
            Fit model
        """
        return self.clf.fit(features,labels)
    
    def predict(self,features_test):
        """
            Predict model
        """
        return self.clf.predict(features_test)
    

# test our model
my_dataset = convert_dataframe_into_dataset(eron_data)
[labels, features] = divide_dataset_into_features_labels(my_dataset,features_list)
features_train, features_test, labels_train, labels_test = train_test_split(features,\
                                                                            labels, test_size=0.3, random_state=42)

## Test our data set and see the performace of it.
clf = DecisionTree(features_train, features_test, labels_train)
print(test_model_tester(clf,my_dataset,features_list));

### Task 5: Create new feature(s)
### Store to my_dataset for easy export below.

eron_data['fraction_messages_to_poi'] = eron_data['from_this_person_to_poi']/eron_data['from_messages']
eron_data['fraction_messages_from_poi'] = eron_data['from_poi_to_this_person']/eron_data['to_messages']

print(eron_data.loc['BANNANTINE JAMES M'])

new_features = ["fraction_messages_from_poi","fraction_messages_to_poi"]

for feature_name in new_features:
    if feature_name  not in features_list:
        features_list.append(feature_name)

### plot new features
fraction_list = ['poi','fraction_messages_from_poi','fraction_messages_to_poi']
data_fraction = eron_data[fraction_list]

data_fraction_poi = data_fraction[data_fraction['poi'] == 1]
data_fraction_non_poi = data_fraction[data_fraction['poi'] == 0]

ax = data_fraction_poi.plot(kind='scatter', x='fraction_messages_from_poi', y='fraction_messages_to_poi',
             color='DarkRed', label='POI', marker="*");
data_fraction_non_poi.plot(kind='scatter', x='fraction_messages_from_poi', y='fraction_messages_to_poi',
             color='DarkBlue', label='Non POI',ax=ax);

plt.ylabel("From this person to Poi")
plt.xlabel('From Poi to this person')
plt.show()

# test our model
my_dataset = convert_dataframe_into_dataset(eron_data)
[labels, features] = divide_dataset_into_features_labels(my_dataset,features_list)
features_train, features_test, labels_train, labels_test = train_test_split(features,\
                                                                            labels, test_size=0.3, random_state=42)

## Test our data set and see the performace of it.
clf = DecisionTree(features_train, features_test, labels_train)
print(test_model_tester(clf,my_dataset,features_list))

## Feature selection 

# Feature selection - selectionkbest, select percentile,lasso regression
selector = SelectKBest(k=5)
features = selector.fit_transform(features,labels)
scores_KB = selector.scores_

print("Final shape of parameters",features.shape)
print("Feature scores:",selector.scores_)
                        
import numpy as np

d = {'feature': features_list[1:], 'score':  scores_KB}
df = pd.DataFrame(data=d)
df = df.sort_values(['score'],ascending=[False])
print(df)

ax = df.plot.bar(xticks=df.index)
ax.set_xticklabels(df.feature)
plt.show()


features_list = ['poi','bonus','salary','fraction_messages_to_poi','total_stock_value','exercised_stock_options']

### Task 6: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Train and test
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)

#Decision Tree
clf = DecisionTree(features_train, features_test, labels_train)
print(test_model_tester(clf,my_dataset,features_list))

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


from sklearn.model_selection import StratifiedKFold,GridSearchCV

def init_cross_validation(train_reduced, labels):
    
    cross_validation = StratifiedKFold(n_splits=10, random_state = 42)
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


def create_decision_tree(train_reduced,labels,run_grid_search=False):
    
    if run_grid_search:
        
        parameter_grid = {
            'max_features': ['sqrt', 'auto', 'log2'],
            'min_samples_split': [2, 3, 10],
            'min_samples_leaf': [1, 2, 3,4, 10]
        }
        clf = DecisionTreeClassifier()
        clf = generate_grid_search_model(clf,parameter_grid,train_reduced,labels).best_estimator_
    else: 
        parameters = {
          'class_weight':None,
          'criterion':'gini', 
          'max_depth':None,
          'max_features':None, 
          'max_leaf_nodes':None,
          'min_impurity_split':1e-07, 
          'min_samples_leaf':1,
          'min_samples_split':2,
          'min_weight_fraction_leaf':0.0,
          'presort':False, 
          'random_state':None, 
           'splitter':'best'
        }
        clf = DecisionTreeClassifier(**parameters)
        clf.fit(train_reduced, labels)

    return clf

clfDT =  create_decision_tree(features,labels,run_grid_search=True)
test_model_tester(clfDT,my_dataset,features_list)    
   

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

tester.dump_classifier_and_data(clfDT, my_dataset, features_list)