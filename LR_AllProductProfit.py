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


Data=pd.read_csv('NumericData1.csv')
Data.rename(columns={'Log_ShiipingCost':'Log_ShippingCost'},inplace=True)

corr = Data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True)
plt.show()

X=Data[['SellingPricePerUnit','PurchasingPricePerUnit','Sales','Quantity','Discount']]
y=Data['Profit']

print("Score Of KFOLD: ")
linreg = linear_model.LinearRegression(normalize=True)
print(linreg)
scores = cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')

mse= -scores

rmse = np.sqrt(mse)
print('RMSE :', rmse.mean())

deg=range(1,5)
for k in deg:
    poly = PolynomialFeatures(degree=k)
    X_deg = poly.fit_transform(X)
    # predict_ = poly.fit_transform(y)
    linreg = linear_model.LinearRegression()
    scores = cross_val_score(linreg, X_deg, y, cv=10, scoring='neg_mean_squared_error')

    mse = -scores

    rmse = np.sqrt(mse)

    print('RMSE for polynomial',k,' :', rmse.mean())

name=('Linear','Polynomial-1','Polynomial-2','Polynomial-3','Polynomial-4')
X_axis=np.arange(len(name))
Y_axis=[98.76,98.76, 1.50,0.34,105.80]
plt.bar(X_axis,Y_axis,align='center',width=.4)
plt.xticks(X_axis,name)
plt.xlabel('Algorithms')
plt.ylabel('RMSE')
plt.title("Performance graph of regression Algorithms(Profit prediction)")
plt.show()


#After Variable tranformation
Data=pd.read_csv('NumericData1.csv')
print('After ariable tranformation:')
print(Data.shape)
X=Data[['SellingPricePerUnit','PurchasingPricePerUnit','Log_Sales','Quantity','Discount']]

y=Data.Profit

sns.boxplot(x=Data['Log_Sales'])
plt.show()

linreg = linear_model.LinearRegression()
scores = cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')

mse= -scores

rmse = np.sqrt(mse)
print('RMSE :', rmse.mean())

deg=range(1,5)
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
Data=Data[Data.Log_Sales<8.69434]
Data=Data[Data.Log_Sales>0.304783]

X=Data[['SellingPricePerUnit','PurchasingPricePerUnit','Log_Sales','Quantity','Discount']]

y=Data.Profit
print(Data.shape)

linreg = linear_model.LinearRegression()
scores = cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error')

mse= -scores

rmse = np.sqrt(mse)
print('RMSE :', rmse.mean())

deg=range(1,5)
for k in deg:
    poly = PolynomialFeatures(degree=k)
    X_deg = poly.fit_transform(X)
    # predict_ = poly.fit_transform(y)
    linreg = linear_model.LinearRegression()
    scores = cross_val_score(linreg, X_deg, y, cv=10, scoring='neg_mean_squared_error')

    mse = -scores

    rmse = np.sqrt(mse)

    print('RMSE for polynomial',k,' :', rmse.mean())


