# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 09:29:46 2021

@author: MHinojosaLee
"""
#Load the libraries
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

data_file_path = "C:/Users/mhinojosalee/Downloads/Machine learning with Python\Exam/"
# Lets read the trining dataset
data_train = pd.read_csv(data_file_path+'train_V2.csv')
# Now we read the training data set
score = pd.read_csv(data_file_path+'score.csv')

print(data_train.shape)
print(data_train.head())
pd.options.display.max_columns = None
print(data_train.describe())


#Visualizing the missing data, percent is the percentage of  of null data by each variable.
total = data_train.isnull().sum().sort_values(ascending=False)
percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)
#table
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
print(data_train['score2_pos'].value_counts())  ##I am not sure why we do this.... do you know???

print(data_train.shape)
data_train = data_train.drop((missing_data[missing_data['Percent'] > 0.15]).index,1)
print(data_train.shape)
print(data_train.head())
#data_train.dropna(inplace=True) #we could drop all that is NaN, but we will loose observations. (4425, 43) instead of (4425, 43). 10 columns are dropped (all the scores)
#I will try imputation
print(data_train.shape)
