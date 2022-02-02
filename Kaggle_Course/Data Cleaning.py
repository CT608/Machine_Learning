# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 21:19:15 2021

@author: CT
"""

import pandas as pd
import numpy as np
# for Box-Cox Transformation
from scipy import stats
# for min_max scaling
from mlxtend.preprocessing import minmax_scaling
# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
# helpful modules
import fuzzywuzzy
from fuzzywuzzy import process
import chardet


data = pd.read_csv("NFL Play by Play 2009-2016 (v3).csv")
pd.set_option('display.max_rows', None)
#print(data.head())

# set seed for reproduction
np.random.seed(0)

#explore data
#print(data.describe())

# 1.MISSING VALUE--------------------------------------------------------------
# 1.1 explore missing value
missing_value_count = data.isnull().sum()
print(missing_value_count[missing_value_count > 0])
total_cells = np.product(data.shape)
percent_missing = round(missing_value_count.sum() / total_cells, 0)
percent_missing_colume = round((missing_value_count / data.shape[0]) * 100, 0)

'''
# 1.2 drop na
# remove all rows with na, default axis=0
data.dropna(axis=0)
# remove all columns with na
data.dropna(axis=1)

# remove a na of a specific column
data.dropna(subset=['Interceptor'])
'''

# 1.3 filling in missing value automaically
'''
data_sub_1 = data.loc[:,'EPA':'Season']
# replace all na with 0
data_sub_1.fillna(0)
# replace all na with the value immediate after it in the same column
# dfill:back fii; ffill: forward fill
data_sub_1.fillna(method = "bfill", axis = 0).fillna(0)

# also you can fill in na with mean, mode, median by calculating the mean, mode, median of each column
x = data_sub_1.EPA.mean()
data_sub_1.fillna(x, axis=0)

'''

# 2 REPLACE DATA---------------------------------------------------------------
# 2.1 replace specific value
'''
df = pd.DataFrame({'one':[10,20,30,40,50,666], 'two':[99,0,30,40,50,60]})
print (df.replace({99:10,666:60,0:20}))
'''

#2.2 cleaning wrong data
'''
# replace data
for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.loc[x, "Duration"] = 120

# remove data    
for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.drop(x, inplace = True)
'''
# 3 DUPLICATE------------------------------------------------------------------
# 3.1 explore duplicates(return true if duplicates)
print(data.duplicated())

# 3.2 remove duplicates
data.drop_duplicates()

# 4 SCALING AND NORMALIZATION--------------------------------------------------
'''
in scaling, you're changing the range of your data, while
in normalization, you're changing the shape of the distribution of your data.
'''
# set seed for reproduction
np.random.seed(0)
data_original = np.random.exponential(size=1000)

# 4.1 scaling
# mix-max scale the data between 0 and 1
data_scaled = minmax_scaling(data_original, columns=[0])

# 4.2 normalisation(((Box-Cox only takes positive values))
data_normalized = stats.boxcox(data_original)

#plotting----------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(10, 5))
sns.histplot(data_original, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(data_scaled, ax=ax[1], kde=True, legend=False)
ax[1].set_title("Scaled data")
sns.histplot(data_normalized, ax=ax[2], kde=True, legend=False)
ax[2].set_title("Normalized data")
plt.show()



# 5 PARSING DATE---------------------------------------------------------------
landslides = pd.read_csv("catalog.csv")
np.random.seed(0)
pd.set_option('display.max_columns', None)
# method 1 : tell python the format
print(landslides.date.head())
landslides["date_parsed"] = pd.to_datetime(landslides.date, format="%m/%d/%y")
print(landslides.date_parsed.head())
# method 2 : ython parsing date itself
landslides["date_parsed2"] = pd.to_datetime(landslides.date, infer_datetime_format=True)
print(landslides.date_parsed2.head())

# selecting day of the month
day_of_month_landslides = landslides.date_parsed.dt.day

# check length of str
date_len = landslides.date.str.len()
date_len_counts = date_len.value_counts()
indices = np.array(np.where(date_len == 8))[0]
landslides.loc[indices, "date"]

"""

# 6 INCONSISTENT DATA ENTRY
professors = pd.read_csv("pakistan_intellectual_capital.csv")
# set seed for reproducibility
np.random.seed(0)
# check country data
countries = professors['Country'].unique()
countries.sort()
print(countries)

#convert to lower case
professors['Country'] = professors['Country'].str.lower()
# remove trailing white spaces
professors['Country'] = professors['Country'].str.strip()


