# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:31:57 2021

@author: CT
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import eli5
from eli5.sklearn import PermutationImportance
import pydotplus

import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import shap
from pdpbox import pdp, get_dataset, info_plots

# Permutation------------------------------------------------------------------
'''
Steps:
1.Get a trained model.

2.Shuffle the values in a single column, make predictions using the resulting dataset. 
Use these predictions and the true target values to calculate how much the loss function 
suffered from shuffling. That performance deterioration measures the importance of the 
variable you just shuffled.

3.Return the data to the original order (undoing the shuffle from step 2). 
Now repeat step 2 with the next column in the dataset, 
until you have calculated the importance of each column.
'''
#load data
data = pd.read_csv("FIFA 2018 Statistics.csv")
#check the string (whether there are mistakes)
print("String in 'Man of the Match' colume: ")
print(data['Man of the Match'].unique())
print("\n")
#set Y and convert it in to binary
y = np.array(data['Man of the Match'] == 'Yes')
# selct X where data types are int64 or float64
x = data.select_dtypes(include = ['int64'])
#check NA
print("NA Column names: ")
print(x.isnull().sum()[x.isnull().sum() != 0])
print("\n")
#slipt train and test date
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
# build model and feed data
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(x_train, y_train)

#permutation inportance calculation
perm = PermutationImportance(model, random_state=1)
perm.fit(x_test, y_test)
eli5.show_weights(perm, feature_names = x_train.columns.tolist())

'''
The values towards the top are the most important features, 
and those towards the bottom matter least.

The first number in each row shows how much model performance decreased 
with a random shuffling

The number after the ± measures how performance varied from one-reshuffling to the next.

You'll occasionally see negative values for permutation importances. 
In those cases, the predictions on the shuffled (or noisy) data 
happened to be more accurate than the real data. 
This happens when the feature didn't matter (should have had an importance close to 0), 
but random chance caused the predictions on shuffled data to be more accurate. 
This is more common with small datasets, like the one in this example, 
because there is more room for luck/chance.
'''

# Partial Plots----------------------------------------------------------------
# show decision tree graph
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(x_train, y_train)
tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=x_train.columns)
graph = pydotplus.graph_from_dot_data(tree_graph)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()
print("done")

'''
1.Leaves with children show their splitting criterion on the top
2.The pair of values at the bottom show the count of False values 
and True values for the target respectively, of data points in that node of the tree.
'''

# Create the data that we will plot
'''
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=x_test, model_features=x_train.columns, feature='Goal Scored')

# plot it
pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show()
'''
'''
1.The y axis is interpreted as change in the prediction from 
what it would be predicted at the baseline or leftmost value.
2.A blue shaded area indicates level of confidence

From this particular graph, we see that scoring a goal substantially 
increases your chances of winning "Man of The Match." But extra goals beyond 
that appear to have little impact on predictions.
'''

#2D Partial Dependence Plots
# Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot
'''
features_to_plot = ['Goal Scored', 'Distance Covered (Kms)']
inter1  =  pdp.pdp_interact(model=tree_model, dataset=x_test, model_features=x_train.columns, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()
'''

#SHAP Values
# randomly choose one row
row_to_show = 1
data_for_prediction = x_test.iloc[row_to_show]
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
model.predict_proba(data_for_prediction_array)

#get shap value
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

#advance use of SHAP values
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values[1], x_test)

#SHAP Dependence Contribution Plots
shap.dependence_plot('Ball Possession %', shap_values[1], x_test, interaction_index="Goal Scored")
