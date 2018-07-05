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

Data=pd.read_csv('DS2.csv',usecols=['Product Name','Quantity ordered new','Discount','Sales','Profit','Unit Price','Product Base Margin','Shipping Cost','Log_Sales','Log_ShiipingCost'])
Data=Data[Data['Product Base Margin']>0]
print(Data.shape)
SinData=Data[Data['Product Name']=='Global High-Back Leather Tilter, Burgundy']
print(SinData.shape)

# corr = SinData.corr()
# sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True)
# plt.show()

X=SinData[['Sales','Quantity ordered new','Discount']]
y=SinData['Profit']

print("Score Of KFOLD: ")
linreg = linear_model.LinearRegression()
print(linreg)
scores = cross_val_score(linreg, X, y, cv=3, scoring='neg_mean_squared_error')

mse= -scores

rmse = np.sqrt(mse)
print('RMSE :', rmse.mean())

deg=range(1,5)
for k in deg:
    poly = PolynomialFeatures(degree=k)
    X_deg = poly.fit_transform(X)
    # predict_ = poly.fit_transform(y)
    linreg = linear_model.LinearRegression()
    scores = cross_val_score(linreg, X_deg, y, cv=3, scoring='neg_mean_squared_error')

    mse = -scores

    rmse = np.sqrt(mse)

    print('RMSE for polynomial',k,' :', rmse.mean())

# RMSE for polynomial 1  : 800.214145555
# RMSE for polynomial 2  : 8132.20605685
# RMSE for polynomial 3  : 6665771.71646
# RMSE for polynomial 4  : 8154381.86474
print('Shipping Cost')
X=SinData[['Unit Price','Product Base Margin']]
y=SinData['Shipping Cost']

print("Score Of KFOLD: ")
linreg = linear_model.LinearRegression()
print(linreg)
scores = cross_val_score(linreg, X, y, cv=3, scoring='neg_mean_squared_error')

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

# RMSE : 0.0
# RMSE for polynomial 1  : 0.0
# RMSE for polynomial 2  : 0.0
# RMSE for polynomial 3  : 0.0
# RMSE for polynomial 4  : 0.0