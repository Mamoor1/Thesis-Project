import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, PolynomialFeatures
import scipy
from scipy.stats.stats import pearsonr


Data=pd.read_csv('NumericData.csv')
SingleProd=Data[Data.ProdID==4640]

X=SingleProd[['Sales','Quantity','Discount']]
y=SingleProd['Profit']

print("Score Of KFOLD: ")
linreg = linear_model.LinearRegression()
scores = cross_val_score(linreg, X, y, cv=3, scoring='neg_mean_squared_error')

mse = -scores

rmse = np.sqrt(mse)
print('RMSE',rmse.mean())
linreg.fit(X,y)
print(linreg.predict([[138.96,3,0]]))
print('Y intercepsts:',linreg.intercept_)
print('X coefficients:',linreg.coef_)

print("Score Of test split: ")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

print('RMSE',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



print("Score Of KFOLD Without Discount: ")
X=SingleProd[['Sales','Quantity']]
y=SingleProd['Profit']
linreg = linear_model.LinearRegression()
scores = cross_val_score(linreg, X, y, cv=3, scoring='neg_mean_squared_error')

mse = -scores

rmse = np.sqrt(mse)
print('RMSE',rmse.mean())


deg=range(1,5)
for k in deg:
    poly = PolynomialFeatures(degree=k)
    X_deg = poly.fit_transform(X)
    # predict_ = poly.fit_transform(y)
    linreg = linear_model.LinearRegression()
    scores = cross_val_score(linreg, X_deg, y, cv=3, scoring='neg_mean_squared_error')

    mse = -scores

    rmse = np.sqrt(mse)
    print('RMSE for polynomial', k, ' :', rmse.mean())