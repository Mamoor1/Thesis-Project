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

SingleProd=Data[Data.ProdID==4640]

X=SingleProd[['Sales','Quantity','OrderPriorityID','ShipModeID','RegionID']]
y=SingleProd['ShippingCost']



deg=range(1,5)
for k in deg:
    poly = PolynomialFeatures(degree=k)
    X_deg = poly.fit_transform(X)
    # predict_ = poly.fit_transform(y)
    linreg = linear_model.LinearRegression()
    scores = cross_val_score(linreg, X_deg, y, cv=4, scoring='neg_mean_squared_error')

    mse = -scores

    rmse = np.sqrt(mse)

    print('RMSE for polynomial',k,' :', rmse.mean())

name=('Polynomial-1','Polynomial-2','Polynomial-3')
X_axis=np.arange(len(name))
Y_axis=[ 3.31,23.18,261.58]
plt.bar(X_axis,Y_axis,align='center',width=.4)
plt.xticks(X_axis,name)
plt.xlabel('Algorithms')
plt.ylabel('RMSE')
plt.title("Performance graph of regression Algorithms(Shipping cost prediction for a particular product)")
plt.show()