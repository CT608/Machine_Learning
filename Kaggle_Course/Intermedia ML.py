# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 11:17:16 2021

@author: CT
"""


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import matplotlib.image as pltimg


# import data
data = pd.read_csv("melb_data.csv")
y = data.Price
melb_predictors = data.drop(['Price'], axis=1)
x = melb_predictors.select_dtypes(exclude=['object'])
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)

# Function for comparing different approaches
def score_dataset(x_train, x_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(x_train, y_train)
    y_preds = model.predict(x_test)
    return mean_absolute_error(y_test, y_preds)

# 1.Deal with missing value----------------------------------------------------

# Method 1: drop missing value

# Method 2: Imputation

imputer = SimpleImputer()
# we only fit x_train, and therefore nas in x_test are replaced with same value as x_train
imputed_x_train = pd.DataFrame(imputer.fit_transform(x_train))
imputed_x_train.columns = x_train.columns
imputed_x_test = pd.DataFrame(imputer.transform(x_test))
imputed_x_test.columns = x_test.columns
print("Imputation MAE is {:.0f}".format(score_dataset(imputed_x_train, imputed_x_test, y_train, y_test)))


#Method 3: An Extension to Imputation
imputer = SimpleImputer(add_indicator=True)
imputed_x_train_plus = pd.DataFrame(imputer.fit_transform(x_train))
imputed_x_test_plus = pd.DataFrame(imputer.transform(x_test))
columns_with_missing = np.array(x_train.columns)
name_missing = np.array(x_train.isnull().sum()[x_train.isnull().sum()>0].index)
for name in name_missing:
    columns_with_missing = np.append(columns_with_missing, name + " missing")
imputed_x_train_plus.columns = columns_with_missing
imputed_x_test_plus.columns = columns_with_missing

print("Extension Imputation MAE is {:.0f}".format(score_dataset(imputed_x_train_plus, imputed_x_test_plus, y_train, y_test)))

# 2.Categorical variable-------------------------------------------------------
# Find categorial columns
# check the columns whose type is object
data = data.dropna(axis = 0)
y = data.Price
x = melb_predictors = data.drop(['Price'], axis=1)
x_no_object = x.select_dtypes(exclude=['object'])

object_columns = x.columns[x.dtypes == "object"]
Categorical_columns = np.array(object_columns[[2, 3, 7]])

x = pd.concat([x[Categorical_columns], x_no_object], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)


# Method 1 drop categorical variable

# Method 2 Ordinal Encoding
ordinal_encode = OrdinalEncoder()
categorical_x_train = x_train.copy()
categorical_x_test = x_test.copy()

categorical_x_train[Categorical_columns] = ordinal_encode.fit_transform(x_train[Categorical_columns])
categorical_x_test[Categorical_columns] = ordinal_encode.transform(x_test[Categorical_columns])
print("Ordinal Encoding MAE is {:.0f}".format(score_dataset(categorical_x_train, categorical_x_test, y_train, y_test)))

# Method 3 One-Hot Encoding  
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_x_train_part = pd.DataFrame(OH_encoder.fit_transform(x_train[Categorical_columns]))
OH_x_test_part = pd.DataFrame(OH_encoder.fit_transform(x_test[Categorical_columns]))

# should give them the same index, or concat will cause mistakes(concat is depend on index)
OH_x_train_part.index = x_train.index
OH_x_test_part.index = x_test.index

OH_x_train = x_train.drop(Categorical_columns, axis=1)
OH_x_test = x_test.drop(Categorical_columns, axis=1)

OH_x_train1 = pd.concat([OH_x_train, OH_x_train_part], axis = 1)
OH_x_test1 = pd.concat([OH_x_test, OH_x_test_part], axis = 1)
print("One-Hot Encoding is {:.0f}".format(score_dataset(OH_x_train, OH_x_test, y_train, y_test)))


