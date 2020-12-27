# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 09:57:34 2020

@author: LENOVO
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

os.chdir('E:\ivy\case 2')
path_data = os.getcwd()
weather_data = pd.read_csv('weatherHistory.csv')
weather_data.head(3)

All_data=weather_data.describe(include='all')# describes allnumerical and  categorical data
Cat_alldata=weather_data.describe(include=['O'])# describes all  categorical data
Cor=weather_data.corr()
dataset_sp=weather_data.iloc[:,[0,2,3,4,5,8]]
Cor_s=dataset_sp.corr()

sns.regplot(x=dataset_sp["Temperature (C)"], y=dataset_sp["Humidity"])

outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers
outlier_data = detect_outlier(dataset_sp["Humidity"])
print (outlier_data)

data_set_clean = dataset_sp[dataset_sp["Humidity"]>0.15]
sns.regplot(x=data_set_clean["Temperature (C)"], y=data_set_clean["Humidity"])

y= data_set_clean.iloc[:,[2]] # Dependent Data
X= data_set_clean.iloc[:,[1,3,4]] #Feature Data
X1= pd.get_dummies(X, columns =['Precip Type'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.coef_
regressor.intercept_

y_pred = regressor.predict(X_test)
y_pred_data=pd.DataFrame(y_pred)


regressor.score(X_train,y_train)
regressor.score(X_test,y_test)

from sklearn import metrics
import math
print(math.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Adding Intercept term to the model
X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)

#Converting into Dataframe
X_train_d=pd.DataFrame(X_train)

#Printing the Model Statistics
model = sm.OLS(y_pred,X_test).fit()
model.summary()



#Checking the VIF Values
from statsmodels.stats.outliers_influence import variance_inflation_factor
[variance_inflation_factor(X_train.values, j) for j in range(X_train.shape[1])]

New_X_train=X_train.drop(["Precip Type_snow"],axis = 1,inplace = True)
New_X_test=X_test.drop(["Precip Type_snow"],axis = 1,inplace = True)

regressor.fit(New_X_train, y_train)

model1 = sm.OLS(y_pred,X_test).fit()
model.summary()


#Storing Coefficients in DataFrame along with coloumn names
coefficients = pd.concat([pd.DataFrame(X_train_d.columns),pd.DataFrame(np.transpose(regressor.coef_))], axis = 1)
