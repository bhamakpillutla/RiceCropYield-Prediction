# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Bhama Pillutla
""" 

import numpy as np
import pandas as pd 
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

from sklearn import preprocessing,svm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LinearRegression
#from sklearn.cross_validation import train_test_split,KFold,cross_val_score,cross_val_predict
import csv

# upload rice crop data

#input file location = C:/datset12.csv
rawdata = 'C:/datset12.csv'

df = pd.read_csv(rawdata)
df.keys()
df.State_Name.shape

print ("DATASET - STATE NAME:",df.State_Name)
print ("DATASET - season:",df.season)


print('Data Set Shape: {0}').format(df.State_Name.shape)
print('Data set columns are {0}').format(df.columns)

#retaining statename, district_name, season, crop, area, and production
df = df[['State_Name','District_Name','season','crop','area','production']]

#data preprocessing
df.dropna(inplace=True)
print('After removing na rows, Data Set Shape: {0}').format(df.State_Name.shape)

#create a dataframe for numerical features
data = pd.DataFrame(df,columns=['area'])
print(data.shape)

#create a dataframe for categorial features
cols_to_transform = pd.DataFrame(df,columns=['State_Name','District_Name','season','crop'])

from sklearn.preprocessing import LabelEncoder
dummies = pd.get_dummies(cols_to_transform)

# Join data1 and dummies using Numpy and yield as array
#setting up independent and dependent variables
X = np.array(data.join(dummies))

y = np.array(df['production'])

print("x values = ",X)
print("y values = ",y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=5)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape) 

def training_testing_phase(regression_model,name):
    # First we fit a model or I am going build a linear regression model using my train test data sets
    regression_model.fit(X_train,y_train)
    pred_train=regression_model.predict(X_train)
    pred_test=regression_model.predict(X_test)
       
    # mean squared error for training and test data
    print("Fit a "+name+" model X_train, and calculate MAE with y_train:",metrics.mean_absolute_error(pred_train,y_train))
    print("Fit a "+name+" X_train, and calculate MAE with X_test, y_test:",metrics.mean_absolute_error(pred_test,y_test))

#training using linear Regression  
lm = LinearRegression()
training_testing_phase(lm,'Linear Regression Model')

#predicting rice crop yield using Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth = 5)
training_testing_phase(dtr,'Decision Tree Regression')


#predicting rice crop yield using KNeighborsRegressor Regressor
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors= 5)
training_testing_phase(knr, 'KNN Regression')
