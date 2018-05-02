import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as m_s_e
import seaborn as sns

data = pd.read_csv("AmesHousing.txt",delimiter="\t")
train = data[0:1460]
test = data[1460:]

#getting information for each column
print(data.info)
target = "SalePrice"


#univariate linear regression!fidning corr
plt.scatter(data['Garage Area'],data['SalePrice'])
plt.xlabel("garage area")
plt.show()
plt.scatter(data['Gr Liv Area'],data['SalePrice'])
plt.xlabel("Gr Liv Area")
plt.show()
plt.scatter(data['Overall Cond'],data['SalePrice'])
plt.xlabel("Overall Cond")
plt.show()


#univariate regression
lr = LinearRegression()
lr.fit(train[['Gr Liv Area']], train['SalePrice'])
predictions_te = lr.predict(test[['Gr Liv Area']])
test_rmse = np.sqrt(m_s_e(predictions_te,test['SalePrice']))

predictions_tr = lr.predict(train[['Gr Liv Area']])
train_rmse = np.sqrt(m_s_e(predictions_tr,train['SalePrice']))

#multivariate regression
cols = ['Overall Cond', 'Gr Liv Area']
#train and do predictions and also calculate rmse for train and test_predictions
lr = LinearRegression()
lr.fit(train[cols], train['SalePrice'])
predictions_te = lr.predict(test[cols])
test_rmse_2 = np.sqrt(m_s_e(predictions_te,test['SalePrice']))

predictions_tr = lr.predict(train[cols])
train_rmse_2 = np.sqrt(m_s_e(predictions_tr,train['SalePrice']))

#handle missing values! let us first get numerical columns in data
numerical_train = train.select_dtypes(include=['int', 'float'])
#drop these columns as of now as they are not directly related
print(numerical_train.columns.values)
#numerical_train = numerical_train.drop(['PID', 'Year Built', 'Year Remod/Add', 'Garage Yr Blt', 'Mo Sold', 'Yr Sold'], axis=1)
null_series = numerical_train.isnull().sum()
full_cols_series = null_series[null_series == 0]
print(full_cols_series)
#let us find out these numerical columns correlate with target cp;i,m
train_subset = train[full_cols_series.index]
temp_Df = train_subset.corr()
temp_Df["SalePrice"] = abs(temp_Df["SalePrice"])
#sorting it by corr values
sorted_corrs = temp_Df["SalePrice"].sort_values()
print(sorted_corrs)
#let us keep our correlation cutoff to 0.3
#also need to find collinearity between features]
#let us utilize seaborn heatmap

strong_corrs = sorted_corrs[sorted_corrs > 0.3]
corrmat = train_subset[strong_corrs.index].corr()
sns.heatmap(corrmat)
"""Based on the correlation matrix heatmap, we can tell that the following pairs of columns are strongly correlated:
Gr Liv Area and TotRms AbvGrd
Garage Area and Garage Cars
we can drop totrms and garage cars and then again try to predict"""

final_corr_cols = strong_corrs.drop(['Garage Cars', 'TotRms AbvGrd'])
features = final_corr_cols.drop(['SalePrice']).index
target = 'SalePrice'
test = test[final_corr_cols.index]
clean_test=test.dropna(axis=0)
print(test.isnull().sum())
lr = LinearRegression()
lr.fit(train[features], train['SalePrice'])

train_predictions = lr.predict(train[features])
test_predictions = lr.predict(clean_test[features])
#removing features with low variance (also rescale features from 0 to 1 using normalization) #bias-variance trade off
unit_train = (train[features] - train[features].min())/(train[features].max() - train[features].min())
#fidning variance
sorted_vars = unit_train.var().sort_values()
print(sorted_vars)
#let us set a cutoff variance of 0.015 and then again predict
clean_test = test[final_corr_cols.index].dropna()
features = features.drop('Open Porch SF')

lr = LinearRegression()
lr.fit(train[features], train['SalePrice'])

train_predictions = lr.predict(train[features])
test_predictions = lr.predict(clean_test[features])

train_mse = m_s_e(train_predictions, train[target])
test_mse = m_s_e(test_predictions, clean_test[target])

train_rmse_2 = np.sqrt(train_mse)
test_rmse_2 = np.sqrt(test_mse)

print(train_rmse_2)

#use gradient descent to minimize the cost function (mse)
