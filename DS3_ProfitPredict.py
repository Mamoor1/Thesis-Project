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


# Data=pd.read_excel('Dataset_3.xlsx',encoding='latin-1')
# print(Data.head())
#
# Data['Log_Sales']=np.log(Data.Sales);
# Data['Log_ShiipingCost']=np.log(Data['Shipping Cost']);
#
# print(Data.head())
#
# Data.set_index('Row ID',inplace=True)
# Data.to_csv('DS2.csv')

import seaborn as sns

Data=pd.read_csv('DS2.csv',usecols=['Quantity ordered new','Discount','Sales','Profit','Unit Price','Log_Quantity','Log_Profit',
                                    'Product Base Margin','Shipping Cost','Log_Sales','Log_ShiipingCost'])
Data.rename(columns={'Log_ShiipingCost':'Log_ShippingCost'},inplace=True)
Data=Data[Data['Product Base Margin']>0]
print(Data.shape)

corr = Data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True)
plt.show()


X=Data[['Sales','Quantity ordered new']]
y=Data['Profit']

print("Score Of KFOLD: ")
linreg = linear_model.LinearRegression()
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


print('Profit pred(Variable tranformation)')

#Data=Data[Data['Profit']>0]
print(Data.shape)
X=Data[['Log_ShippingCost','Log_Sales','Log_Quantity']]
y=Data['Profit']

print("Score Of KFOLD: ")
linreg = linear_model.LinearRegression()
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


sns.boxplot(x=Data['Log_Quantity'])
plt.show()
sns.boxplot(x=Data['Log_Sales'])
plt.show()
sns.boxplot(x=Data['Log_ShippingCost'])
plt.show()
print('After outlier removal:')
print(Data.shape)
Data=Data[Data.Log_ShippingCost<4.8533]
Data=Data[Data.Log_Sales<10.3783]
Data=Data[Data.Log_Sales>0.489301]
Data=Data[Data.Log_Quantity<4.40]
Data=Data[Data.Log_Quantity>0.6861]
print(Data.shape)


X=Data[['Log_ShippingCost','Log_Sales','Log_Quantity']]
y=Data['Profit']

print("Score Of KFOLD: ")
linreg = linear_model.LinearRegression()
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

# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
# RMSE : 913.082968298
# RMSE for polynomial 1  : 913.082968298
# RMSE for polynomial 2  : 924.981744191
# RMSE for polynomial 3  : 961.37794287
# RMSE for polynomial 4  : 2313.88149564
# Profit pred(Variable tranformation)
# Score Of KFOLD:
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
# RMSE : 1.38788496237
# RMSE for polynomial 1  : 1.38788496237
# RMSE for polynomial 2  : 1.37038986275
# RMSE for polynomial 3  : 1.34374371336
# RMSE for polynomial 4  : 1.33355527304
# After outlier removal:
# (4822, 11)
# (4633, 11)
# Score Of KFOLD:
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
# RMSE : 1.3704264669
# RMSE for polynomial 1  : 1.3704264669
# RMSE for polynomial 2  : 1.343148707
# RMSE for polynomial 3  : 1.32193353338
# RMSE for polynomial 4  : 1.34608442205