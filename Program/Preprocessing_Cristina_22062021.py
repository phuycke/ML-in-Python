# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 09:29:46 2021

@author: MHinojosaLee
"""
# %%import the libraries
import numpy as np
import pandas as pd
from matplotlib import animation as ani, pyplot as plt
import seaborn as sns #pretty graphics R style

from IPython.display import HTML
from sklearn.inspection import permutation_importance
plt.style.use('seaborn-darkgrid')
from sklearn.inspection import permutation_importance
import matplotlib as mpl 
import matplotlib.pyplot as plt #graphics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler #library for the rescaling
import statsmodels.api as sm 
import statsmodels.formula.api as smf
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from treeinterpreter import treeinterpreter as ti, utils
import joblib
# %% Load data
data_file_path = "C:/Users/mhinojosalee/Downloads/Machine learning with Python\Exam/"
# Lets read the trining dataset
data_train = pd.read_csv(data_file_path+'train_V2.csv')
# Now we read the training data set
score = pd.read_csv(data_file_path+'score.csv')

print(data_train.shape)
print(data_train.head())
pd.options.display.max_columns = None
print(data_train.describe())


# %%Visualizing the missing data, percent is the percentage of  of null data by each variable.
# total = data_train.isnull().sum().sort_values(ascending=False)
# percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)
# (data_train.isnull().sum(axis=1))[data_train.isnull().sum(axis=1) > 30]
# table
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(30))
# print(data_train['score2_pos'].value_counts())  ##I am not sure why we do this.... do you know???


# data_train = data_train.drop((missing_data[missing_data['Percent'] > 0.30]).index,1)
print(data_train.shape)
print(data_train.head())
# data_train.dropna(inplace=True) #we could drop all that is NaN, but we will loose observations. (4425, 43) instead of (4425, 43) 
# %%Remove outcomes

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
# %%Mode imputer instead of separate categories
# I started with the separated categories, but then I decided a mode imputer for NaN
# datafull['client_segment'] = pd.Categorical(datafull['client_segment'])
# datafull['sect_empl'] = pd.Categorical(datafull['sect_empl'])
# datafull['retired'] = pd.Categorical(datafull['retired'])
# datafull['gold_status'] = pd.Categorical(datafull['gold_status'])
# datafull['prev_stay'] = pd.Categorical(datafull['prev_stay'])
# datafull['divorce'] = pd.Categorical(datafull['divorce'])


impute_mode = SimpleImputer (strategy='most_frequent')
for cols in ['client_segment', "credit_use_ic", "gluten_ic", "lactose_ic","insurance_ic","marketing_permit", "presidential", "urban_ic", "prev_all_in_stay", "shop_use", 
             "company_ic", "dining_ic", "spa_ic","sport_ic","empl_ic",'sect_empl', "retired", "gold_status", "prev_stay", 'divorce', "gender"]:  
      datafull[cols] = impute_mode.fit_transform(datafull[[cols]])

# %%Dummify them!
datafull['client_segment'] = pd.Categorical(datafull['client_segment'])
datafull['sect_empl'] = pd.Categorical(datafull['sect_empl'])
pd.get_dummies(datafull[['client_segment', 'sect_empl']], dummy_na=False).head()
print(datafull.shape)
datafull2 = pd.concat([datafull,pd.get_dummies(datafull[['gender','client_segment', 'sect_empl']], dummy_na=False)], axis=1)
print(datafull2.shape)
print(datafull2.head(1000))

# %%Dropping the original features and one dummy.
print(datafull2.shape)
datafull2.drop(['client_segment', 'sect_empl', 'gender', 'client_segment_5.0','sect_empl_6.0','gender_V'], axis=1, inplace=True)
print(datafull2.shape)

datafull2['profitpernight'] = datafull2['profit_am'] / datafull2['nights_booked']

# %%Missing values per column. I decided to not drop them and use a mean imput instead. 
# During class it was mentioned that sometimes it was not worthy, but I wanted to see. Specially because there were the scores that were giving troubles.
# The scores are quantitative.

impute_quant = SimpleImputer (strategy='mean')
for cols in ['score1_pos', 'score1_neg', 'score2_pos', 'score2_neg', 'score3_pos',
       'score3_neg', 'score4_pos', 'score4_neg', 'score5_pos', 'score5_neg']:  # Missing data, Scores are quantitative
      datafull2[cols] = impute_quant.fit_transform(datafull2[[cols]])


# %%Missing values per row: 
print(datafull2.shape)
datafull2.dropna(thresh = datafull2.shape[1]*0.3, axis = 0, inplace = True)
print(datafull2.shape)
# And here we find that there are not missing values from the rows. So we go to imputting the rest of the missing values

# %%Imputation using the mean for the other missing values
print(datafull2.isnull().sum().sum())
datafull2.fillna(datafull2.mean(), inplace=True)
print(datafull2.isnull().sum().sum())

# %% Time to the rescale
scaler = StandardScaler()
datafull3 = pd.DataFrame(scaler.fit_transform(datafull2))
datafull3.columns = datafull2.columns

# %% Now the test and the train sets should be separated again (they were together just for the preprocessing so that what we do for one, we do for the other one)
data_train = pd.concat([data_train[['outcome_profit', 'outcome_damage_inc', 'outcome_damage_amount']],datafull3[0:5000]], axis=1)
print(data_train.shape)
score = datafull3[5000:5500] #The score dataset will be the last 500 observations
score.shape
# Now it is the time to split the train and the test data sets. We decided to use 20% for testing and 80% for training
X_train, X_test, y_train, y_test = train_test_split(data_train.drop(['outcome_profit', 'outcome_damage_inc', 'outcome_damage_amount'],1),
                                                    data_train['outcome_profit'], test_size=0.2, random_state=48)

# %%To answer the 2nd question, we are going to use Gradient Boost. During class it was discussed that it was a very good algorithm.
# Our model will try 500 random hyperparameter combinations, each time using 5 Cross Validation folds, totalling 2500 fits
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 50)]
learning_rate = [x for x in np.logspace(start = -3, stop = -0.01, num = 50)]
max_features = ['auto']
max_depth = [int(x) for x in np.linspace(1, 10, num = 10)]
min_samples_split = [2, 5, 10, 30]
min_samples_leaf = [1, 2, 4, 10, 30]
subsample = [0.4, 0.6, 0.8, 1]
random_grid = {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'subsample': subsample}
gbm = GradientBoostingRegressor()
gbm_random = RandomizedSearchCV(estimator = gbm, param_distributions = random_grid, n_iter = 500, cv = 5, verbose=2, random_state=42, n_jobs = -1)
gbm_random.fit(X_train, y_train)
gbm_random.best_params_

# %%It took around 30 to 40 minutes to run the model. So we will pickle it to not lose it.
joblib.dump(gbm_random, 'random_search_gbm.pkl')

# %%Output overview of the random search, if using spyder, this can be done in the console.
pd.DataFrame(gbm_random.cv_results_).head()

# %%now We will inspect the best hyperparameter combination. This boasts an average test score of 0.789
pd.DataFrame(gbm_random.cv_results_).loc[pd.DataFrame(gbm_random.cv_results_)['mean_test_score'].idxmax()]

# %% Now, we will fit the model for profit as asked in question 2. For this, we are using the best hyperparameters.
params = gbm_random.best_params_
gbm_profit = GradientBoostingRegressor(**params)
gbm_profit.fit(X_train, y_train)
# I got R2: 0.953 for the X_train and R2: 0.822 for the X_test
print('R2: %.3f' % gbm_profit.score(X_train, np.array(y_train).reshape(-1,1)))
print('R2: %.3f' % gbm_profit.score(X_test, np.array(y_test).reshape(-1,1))) #Here we are using the "holdout" set already

# %% And now we are scoring the 500 potential customers with it
profit_preds = gbm_profit.predict(score)


# %%  A try to "whiteboxing":
# We were interested in identifying what were the variables that mattered, so we used Variable importances based on impurity reduction.

gbm_profit.feature_importances_.sum()
d = {'feature':X_train.columns, 'importance':gbm_profit.feature_importances_}
importances = pd.DataFrame(data=d)
importances.sort_values('importance', ascending=False,inplace=True)

plt.rcdefaults()
plt.rcParams['figure.figsize'] = (4, 3)
fig, ax = plt.subplots()
variables = importances.feature
y_pos = np.arange(len(variables))
scaled_importance = importances.importance
ax.barh(y_pos, scaled_importance, align='center', color='deepskyblue', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.invert_yaxis()
ax.set_xlabel('Scaled Importance')
ax.set_title('Variable Importance')
plt.show()
# This graphic is unorganized. All the variables are shown in the y axis, it is difficult to read them. So Next, I will organize it.

# %%Last plot is not ordered,so the next part is to order it. 
importances2 = importances.copy()
importances2 = importances2.head(20)
import matplotlib.pyplot as plt
plt.rcdefaults()
plt.rcParams['figure.figsize'] = (4, 3)
fig, ax = plt.subplots()
variables = importances2.feature
y_pos = np.arange(len(variables))
scaled_importance = importances2.importance
ax.barh(y_pos, scaled_importance, align='center', color='deepskyblue', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.invert_yaxis()
ax.set_xlabel('Scaled Importance')
ax.set_title('Variable Importance')
plt.show()
# And now we have a nice plot
# %%Permutation importance: As part of the whitening try:
imp = permutation_importance(gbm_profit, X_train, y_train,n_repeats=10,
                                random_state=42, n_jobs=2)

sorted_idx = imp.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(imp.importances[sorted_idx][48:58].T,
           vert=False, labels=X_train.columns[sorted_idx][48:58])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()
# Feature importance and Permutation importance identify the same 2 strongly predictive features for our model for Profit: Profit and Nights booked, which makes sense