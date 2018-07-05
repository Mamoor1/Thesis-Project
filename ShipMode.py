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
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

Data=pd.read_csv('NumericData.csv')

# X=Data[['ProdID','Sales','Quantity','Discount','Profit','ShippingCost','CityID','RegionID',
#                                         'SubCategoryID','OrderPriorityID','CategoryID','SellingPricePerUnit','PurchasingPricePerUnit','DateDif','ShipModeID','MarketID','StateID','CountryID']]
# y=Data['ShipModeID']
#
# logreg = LogisticRegression()
#
# rfecv = RFECV(estimator=logreg, step=1, cv=StratifiedKFold(2),
#               scoring='accuracy')
# rfecv.fit(X, y)
#
# print("Optimal number of features : %d" % rfecv.n_features_)
# print(rfecv.ranking_)
#
# plt.figure()
# plt.xlabel("Number of features selected(Ship Mode prediction)")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()

sns.boxplot(x=Data['DateDif'])
plt.show()


logreg = LogisticRegression(max_iter=100,tol=0.0001)
print(logreg)
X=Data[['DateDif']]
y=Data['ShipModeID']

knn = KNeighborsClassifier(n_neighbors=35)
print(knn)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores.mean())


scores = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
print(scores.mean())

name=('KNN with K=35','Logistic')
X_axis=np.arange(len(name))
Y_axis=[81.40,78.11]
plt.bar(X_axis,Y_axis,align='center',width=.2)
plt.xticks(X_axis,name)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy in %')
plt.title("Perf. graph of Classification Algorithms(Shipping Mode prediction)")
plt.show()