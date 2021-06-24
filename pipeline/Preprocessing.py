# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
# ---

# %%
"""
Created on Fri Jun 18 09:29:46 2021

@author: MHinojosaLee
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 09:29:46 2021

@author: MHinojosaLee
"""
# %% Load the libraries
import numpy as np
import pandas as pd
from matplotlib import animation as ani, pyplot as plt
import seaborn as sns #pretty graphics R style

from IPython.display import HTML

plt.style.use('seaborn-darkgrid')

import matplotlib as mpl 
import matplotlib.pyplot as plt #graphics

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler #library for the rescaling
import statsmodels.api as sm 
import statsmodels.formula.api as smf
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
import getpass
from pathlib import Path
# %% Load data
if getpass.getuser() == 'daniel':
      data_file_path = Path("/home/daniel/PhD/Projects/ML-in-Python")
      # Lets read the trining dataset
      data_train = pd.read_csv(data_file_path / 'data' / 'train_V2.csv')
      # Now we read the training data set
      score = pd.read_csv(data_file_path/  'data' / 'score.csv')
else:
      data_file_path = "C:/Users/mhinojosalee/Downloads/Machine learning with Python\Exam/"
      # Lets read the trining dataset
      data_train = pd.read_csv(data_file_path+'train_V2.csv')
      # Now we read the training data set
      score = pd.read_csv(data_file_path+'score.csv')

print(data_train.shape)
print(data_train.head())
pd.options.display.max_columns = None
print(data_train.describe())


# %% Visualizing the missing data, percent is the percentage of  of null data by each variable.
#total = data_train.isnull().sum().sort_values(ascending=False)
#percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)
#(data_train.isnull().sum(axis=1))[data_train.isnull().sum(axis=1) > 30]
#table
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(30))
#print(data_train['score2_pos'].value_counts())  ##I am not sure why we do this.... do you know???


#data_train = data_train.drop((missing_data[missing_data['Percent'] > 0.30]).index,1)
print(data_train.shape)
print(data_train.head())
#data_train.dropna(inplace=True) #we could drop all that is NaN, but we will loose observations. (4425, 43) instead of (4425, 43) 
# %% Remove outcomes

print(data_train.columns)
data_feat = data_train.drop(['outcome_profit', 'outcome_damage_inc', 'outcome_damage_amount'], axis=1)
print(data_feat.shape)

# %% fuse the score and the training data
print(score.shape)
datafull = pd.concat([data_feat, score])
print(datafull.shape)
print(datafull['client_segment'].value_counts())
print(datafull['sect_empl'].value_counts())
print(datafull['gender'].value_counts())
print(datafull['retired'].value_counts())
print(datafull['gold_status'].value_counts())
print(datafull['prev_stay'].value_counts())
print(datafull['divorce'].value_counts())
print(datafull['married_cd'].value_counts())

#I started with the separated categories, but then I decided a mode imputer for NaN
#datafull['client_segment'] = pd.Categorical(datafull['client_segment'])
#datafull['sect_empl'] = pd.Categorical(datafull['sect_empl'])
#datafull['retired'] = pd.Categorical(datafull['retired'])
#datafull['gold_status'] = pd.Categorical(datafull['gold_status'])
#datafull['prev_stay'] = pd.Categorical(datafull['prev_stay'])
#datafull['divorce'] = pd.Categorical(datafull['divorce'])

# %% Mode imputer instead of separate categories... married_ic gave me problems. I used what he mentioned in class, using mode for categoricals, and mean for the rest. 
#Married_ic is a bolean

impute_mode = SimpleImputer (strategy='most_frequent')
for cols in ['client_segment', "credit_use_ic", "gluten_ic", "lactose_ic","insurance_ic","marketing_permit", "presidential", "urban_ic", "prev_all_in_stay", "shop_use", 
             "company_ic", "dining_ic", "spa_ic","sport_ic","empl_ic",'sect_empl', "retired", "gold_status", "prev_stay", 'divorce', "gender"]:  
      datafull[cols] = impute_mode.fit_transform(datafull[[cols]])
print(datafull.shape)
print (datafull.head())
print(datafull.columns)
datafull = pd.concat([datafull,pd.get_dummies(datafull[['gender']])], axis=1)

# %% Missing values per column
(datafull.isnull().mean())[datafull.isnull().mean() > 0.30]
print(datafull.shape)
datafull.dropna(thresh = datafull.shape[0]*0.3, axis = 1, inplace = True)
print(datafull.shape)

# %% Missing values per row
print(datafull.shape)
datafull.dropna(thresh = datafull.shape[1]*0.3, axis = 0, inplace = True)
print(datafull.shape)

# %% Imputation: this time with the mean
print(datafull.isnull().sum().sum())
datafull.fillna(datafull.mean(), inplace=True)
print(datafull.isnull().sum().sum())

# %% Scaling
scaler = StandardScaler()
datafull2 = pd.DataFrame(scaler.fit_transform(datafull))
datafull2.columns = datafull.columns
#And here I have an error... It is caused by having gender not coded as 0 and 1
