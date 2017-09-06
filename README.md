# Udacity_MachineLearing

This is a project for the Machine Learning Module in Udacity Data Analysis Nonodegree. The aim of this project was to classify the POI in Eron Dataset. 

In this project I perform some outlier removal, feature engineering, feature selection and also tested with different models,tuned the model and also did some validation and evaluation.

## Context 

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.


## To run 

It should run the script poi_id.py and then tester.py to evaluate our model. This project was done with python2.7 so you may also need to use conda in order to run this project.

### Questions 

1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those? [relevant rubric items: “data exploration”, “outlier investigation”]

	The goal of this project is to succesfully identify the POIs of the Eron scandal. In order to do it I will use machine learning classify algorithm, that can throught the data identify patterns of the POI. In this project I will use the dataset contaning some financial data and email data.

2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values. [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

	I end up using in my POI identifier the features ['fraction_messages_to_poi','salary','total_stock_value','bonus','exercised_stock_options'] to select them I used the SelectKBest with k=5. I used scalling before the feature selection. I used selection in order to better scale the financial features. I created two new variables fraction of emails received from POI and fraction of emails sent to poi.


3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms? [relevant rubric item: “pick an algorithm”]

	I tried to use knn, naive bayes, decision trees and some ensembles classifiers like adaboost and random forest. I noticed that the perfomace differ from one model to another and with the decision tree it had the best peformace, with Accuracy: 0.89740 Precission: 0.61636 Recall: 0.61050 I also noticed that with AdaBoost I had a great metrics results as well.


4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well? How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier). [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]


	Tuning the algorithm means finding the best parameters for a certain model, that gives a better results. I tuned my particular algorithm for the Decision Tree and got {'max_features': 'auto', 'min_samples_split': 3, 'max_depth': 6, 'min_samples_leaf': 3} parameters. For my Adaboost model I tuned the n_estimators. And the final result was with {'n_estimators': 130} parameter.


5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis? [relevant rubric items: “discuss validation”, “validation strategy”]

	Validation process is when you divided your dataset into train, test datasets. A classic mistake is to not divide into these to dataset and use the entire data to train your model. I validade my model using a croess-validation (that means dividing my dataset into small chucks and using most of them to train and a chuck to test and then repeat the process).To validate my model I used a  k-fold = 5.

6. Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

	And also to validate I used the accuracy, recall and prediction score. Accuracy is the metric for the pecentage of correctness of the model. Recall is the fraction that have been retrieved over the total amount of relevant instances. And Precision is the  fraction of relevant instances among the retrieved instances,

	In my final model I got Accuracy Score: 89.4% Recall Score: 61.4% Precision Score: 55.2%.

