# Simple Linear Regression

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/admin/Documents/machine learning/TSLA.csv')
dataset.columns
X = dataset.iloc[:,[1,3,4]]
y = dataset['High']
X.columns
y

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


regressor = LinearRegression()
regmodel=regressor.fit(X_train, y_train)
regmodel
regressor.coef_

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred

mean_squared_error(y_test,y_pred)
r2_score(y_test,y_pred)

y_pred=regressor.predict([[282,270.100006,276.540009]])
y_pred

