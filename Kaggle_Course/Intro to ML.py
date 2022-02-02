# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 00:18:54 2021

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
import matplotlib.pyplot as plt
import matplotlib.image as pltimg


# import data
data = pd.read_csv("melb_data.csv")

# explore data
pd.set_option('display.max_columns', None)
print(data.describe())
print(data.columns)

# drop missing value
data = data.dropna(axis = 0)

# selecting data for modeling

#selecting prediction target
y = data.Price
# choosing features
data_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = data[data_features]
print(x.describe())
print(x.head())

# building model
data_model = DecisionTreeRegressor(random_state = 1)  #Many machine learning models allow some randomness in model training. Specifying a number for random_state ensures you get the same results in each run. 
data_model.fit(x, y)

# making prediction
print("Prediction of first five row")
print(data_model.predict(x.head()))

# model validation
predicted_home_price = data_model.predict(x)
print(mean_absolute_error(y, predicted_home_price))

#training and testing data
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state = 0)
data_model = DecisionTreeRegressor(random_state = 1) 
data_model.fit(train_x, train_y)
predict_y = data_model.predict(test_x)
print(mean_absolute_error(test_y, predict_y))

# check overfitting and underfitting
#construct function to compare MAE of different number of lead nodes
def get_mae(max_leaf_nodes, train_x, train_y, test_x, test_y):
    model = DecisionTreeRegressor(random_state = 1, max_leaf_nodes = max_leaf_nodes)
    model.fit(train_x, train_y)
    preds_y = model.predict(test_x)
    mae = mean_absolute_error(test_y, preds_y)
    return(mae)
# apply defined function to test model with different max number of leaf nodes
for i in [5, 50, 500, 5000, 50000]:
    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state = 0)
    mae = get_mae(i, train_x, train_y, test_x, test_y)
    print("Max number of leaf: " , i, "   Mean absolute error: ", round(mae, 0))

# Random forests
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state = 0)
data_model_forest = RandomForestRegressor()
data_model_forest.fit(train_x, train_y)
predict_forest_y = data_model_forest.predict(test_x)
print("Random Forests MAE: ", mean_absolute_error(test_y, predict_forest_y))

# Decision tree flow chart(too large do not use)
'''
dtree = DecisionTreeClassifier()
dtree.fit(test_x, test_y)
data2 = tree.export_graphviz(dtree, out_file=None, feature_names=data_features)
graph = pydotplus.graph_from_dot_data(data2)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()
print("done")
'''




