

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import os
os.chdir('E:\ivy\case 1')                 
path_data = os.getcwd()
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values #Feature Data
y = dataset.iloc[:, 4].values # Dependent Data
X_data=pd.DataFrame(X)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [3])], remainder='passthrough')

X = ct.fit_transform(X)
X2=pd.DataFrame(X)
X= np.array(X, dtype=float)


dataset.isnull().sum()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred_data=pd.DataFrame(y_pred)

regressor.score(X_train,y_train)

regressor.score(X_test,y_test)
import statsmodels.api as sm
model = sm.OLS(y_pred,X_test).fit()
model.summary()

X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)

#Converting into Dataframe
X_train_d=pd.DataFrame(X_train)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] =[variance_inflation_factor(X_train_d.values, j) for j in range(X_train_d.shape[1])]
vif["features"] = X_train_d.columns
vif.round(1)

#Storing Coefficients in DataFrame along with coloumn names
coefficients = pd.concat([pd.DataFrame(X_train_d.columns),pd.DataFrame(np.transpose(regressor.coef_))], axis = 1)











