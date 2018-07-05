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

Data=pd.read_csv('NumericData.csv')

X=Data[['Sales']]
y=Data['ShippingCost']


deg=range(1,6)
for k in deg:
    poly = PolynomialFeatures(degree=k)
    X_deg = poly.fit_transform(X)
    # predict_ = poly.fit_transform(y)
    linreg = linear_model.LinearRegression()
    scores = cross_val_score(linreg, X_deg, y, cv=10, scoring='neg_mean_squared_error')

    mse = -scores

    rmse = np.sqrt(mse)

    print('RMSE for polynomial',k,' :', rmse.mean())

name=('Polynomial-1','Polynomial-2','Polynomial-3','Polynomial-4','Polynomial-5')
X_axis=np.arange(len(name))
Y_axis=[29.26,26.27,26.74,30.32,89.53]
plt.bar(X_axis,Y_axis,align='center',width=.4)
plt.xticks(X_axis,name)
plt.xlabel('Algorithms')
plt.ylabel('RMSE')
plt.title("Performance graph of regression Algorithms(Shipping cost prediction for any products")
plt.show()

#Improved Performance

Data=pd.read_csv('NumericData1.csv',usecols=['Sales','Quantity','OrderPriorityID','ShipModeID','Profit',
                                             'RegionID','ShippingCost','Log_Sales','Log_ShiipingCost','Log_Profit','Log_Quantity'])

corr = Data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True)
plt.show()
print(Data.shape)
# X=Data[['Log_Sales','Quantity','OrderPriorityID','ShipModeID','RegionID']]
X=Data[['Log_Sales']]
y=Data['ShippingCost']


deg=range(1,6)
for k in deg:
    poly = PolynomialFeatures(degree=k)
    X_deg = poly.fit_transform(X)
    # predict_ = poly.fit_transform(y)
    linreg = linear_model.LinearRegression()
    scores = cross_val_score(linreg, X_deg, y, cv=10, scoring='neg_mean_squared_error')

    mse = -scores

    rmse = np.sqrt(mse)

    print('RMSE for polynomial',k,' :', rmse.mean())

print('After Outlier removal')
sns.boxplot(x=Data['Log_Quantity'])
plt.show()
sns.boxplot(x=Data['Log_Sales'])
plt.show()


Data = Data[Data.Log_Sales < 8.69434]
Data = Data[Data.Log_Sales > 0.304783]

X=Data[['Log_Sales']]
y=Data['ShippingCost']
print(Data.shape)

linreg = linear_model.LinearRegression()
scores = cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')

mse = -scores

rmse = np.sqrt(mse)
print('RMSE :', rmse.mean())

deg = range(1, 8)
for k in deg:
        poly = PolynomialFeatures(degree=k)
        X_deg = poly.fit_transform(X)
        # predict_ = poly.fit_transform(y)
        linreg = linear_model.LinearRegression()
        scores = cross_val_score(linreg, X_deg, y, cv=10, scoring='neg_mean_squared_error')

        mse = -scores

        rmse = np.sqrt(mse)

        print('RMSE for polynomial', k, ' :', rmse.mean())


# Data=Data[Data.Log_Sales<8.6462]
# Data=Data[Data.Log_Sales>0.280744]
#
# print(Data.shape)
#
# X=Data[['Log_Sales','Quantity','OrderPriorityID','ShipModeID','RegionID']]
# y=Data['Log_ShiipingCost']
#
# sns.boxplot(x=Data['Log_Sales'])
# plt.show()
#
# deg=range(1,6)
# for k in deg:
#     poly = PolynomialFeatures(degree=k)
#     X_deg = poly.fit_transform(X)
#     # predict_ = poly.fit_transform(y)
#     linreg = linear_model.LinearRegression()
#     scores = cross_val_score(linreg, X_deg, y, cv=10, scoring='neg_mean_squared_error')
#
#     mse = -scores
#
#     rmse = np.sqrt(mse)
#
#     print('RMSE for polynomial',k,' :', rmse.mean())