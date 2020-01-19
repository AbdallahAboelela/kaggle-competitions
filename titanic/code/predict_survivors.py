# Abdallah Aboelela
# Titanic Survivors Kaggle Competition
# https://www.kaggle.com/c/titanic/data
# January 19, 2020

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os

def read_data(file_path, ind_var):
    training = pd.read_csv(file_path + '/train.csv')

    cols = training.columns

    y = training[ind_var]
    X = training.loc[:, training.columns != ind_var]
    X = X.drop(['Cabin', 'Ticket', 'Name'], axis = 1)

    testing = pd.read_csv(file_path + '/test.csv')
    testing = testing.drop(['Cabin', 'Ticket', 'Name'], axis = 1)

    X["Age"] = X["Age"].astype(float)
    testing["Age"] = testing["Age"].astype(float)

    for col in ["Embarked", "Sex"]:
            X[col] = X[col].fillna("miss")
            testing[col] = testing[col].fillna("miss")

            le = preprocessing.LabelEncoder()
            le.fit(X[col])

            X[col] = le.transform(X[col])
            testing[col] = le.transform(testing[col])

    for col in X.columns:
        if col not in ["Embarked", "Sex"]:
            X[col + '_miss'] = np.isnan(X[col])
            X[col] = X[col].fillna(0)

            testing[col + '_miss'] = np.isnan(testing[col])
            testing[col] = testing[col].fillna(0)

    return X, y, testing

def train_and_predict(X, y, testing):
    clf = LogisticRegression(random_state=0).fit(X, y)
    prediction = clf.predict(testing)

    return prediction

def run():
    file_path = '../raw_data'
    ind_var = 'Survived'

    X, y, testing = read_data(file_path, ind_var)

    prediction = train_and_predict(X, y, testing)

    d = {'PassengerID': testing["PassengerId"], 'Survived': prediction}
    final = pd.DataFrame(data = d)

    final.to_csv('../output/prediction.csv', index = False)