import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants.constants import alpha
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import scipy
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import seaborn as sns

Data=pd.read_csv('DS2.csv',usecols=['Quantity ordered new','Discount','Sales','Profit','Unit Price','Product Base Margin','Shipping Cost','Log_Sales','Log_ShiipingCost'])
Data=Data=Data[Data['Product Base Margin']>0]
print(Data.head())

#Shipping cost prediction

#'Quantity ordered new','Discount','Sales','Profit','Unit Price','Product Base Margin','Shipping Cost','Log_Sales','Log_ShiipingCost'
print('Shipping Cost prediction :')
X=Data[['Sales','Product Base Margin']]
y=Data['Shipping Cost']


print("Score Of KFOLD: ")
linreg = linear_model.LinearRegression()
print(linreg)
scores = cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')

mse= -scores

rmse = np.sqrt(mse)
print('RMSE :', rmse.mean())

deg=range(1,8)
for k in deg:
    poly = PolynomialFeatures(degree=k)
    X_deg = poly.fit_transform(X)
    # predict_ = poly.fit_transform(y)
    linreg = linear_model.LinearRegression()
    scores = cross_val_score(linreg, X_deg, y, cv=10, scoring='neg_mean_squared_error')

    mse = -scores

    rmse = np.sqrt(mse)

    print('RMSE for polynomial',k,' :', rmse.mean())

#Shipping Cost pred(Variable tranformation)
print('Shipping Cost pred(Variable tranformation)')

X=Data[['Log_Sales','Product Base Margin']]
y=Data['Shipping Cost']

print("Score Of KFOLD: ")
linreg = linear_model.LinearRegression()
print(linreg)
scores = cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')

mse= -scores

rmse = np.sqrt(mse)
print('RMSE :', rmse.mean())

deg=range(1,8)
for k in deg:
    poly = PolynomialFeatures(degree=k)
    X_deg = poly.fit_transform(X)
    # predict_ = poly.fit_transform(y)
    linreg = linear_model.LinearRegression()
    scores = cross_val_score(linreg, X_deg, y, cv=10, scoring='neg_mean_squared_error')

    mse = -scores

    rmse = np.sqrt(mse)

    print('RMSE for polynomial',k,' :', rmse.mean())


#Outlier removal


sns.boxplot(x=Data['Log_Sales'])
plt.show()

#10.3783 & 0.489301
sns.boxplot(x=Data['Product Base Margin'])
plt.show()
print(Data.shape)
Data=Data[Data.Log_Sales<10.3783]
Data=Data[Data.Log_Sales>0.489301]
print(Data.shape)

# sns.boxplot(x=Data['Log_Sales'])
# plt.show()
# sns.boxplot(x=Data['Product Base Margin'])
# plt.show()

X=Data[['Log_Sales','Product Base Margin']]
y=Data['Shipping Cost']

print("Score Of KFOLD: ")
linreg = linear_model.LinearRegression()
print(linreg)
scores = cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')

mse= -scores

rmse = np.sqrt(mse)
print('RMSE :', rmse.mean())

deg=range(1,8)
for k in deg:
    poly = PolynomialFeatures(degree=k)
    X_deg = poly.fit_transform(X)
    # predict_ = poly.fit_transform(y)
    linreg = linear_model.LinearRegression()
    scores = cross_val_score(linreg, X_deg, y, cv=10, scoring='neg_mean_squared_error')

    mse = -scores

    rmse = np.sqrt(mse)

    print('RMSE for polynomial',k,' :', rmse.mean())


