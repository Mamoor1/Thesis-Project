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

X=Data[['ProdID','Sales','Quantity','Discount','Profit','ShippingCost','CityID','RegionID',
                                        'SubCategoryID','CategoryID','SellingPricePerUnit','PurchasingPricePerUnit','DateDif','ShipModeID','MarketID','StateID','CountryID']]
y=Data['OrderPriorityID']

logreg = LogisticRegression()

# rfecv = RFECV(estimator=logreg, step=1, cv=StratifiedKFold(2),
#               scoring='accuracy')
# rfecv.fit(X, y)
#
# print("Optimal number of features : %d" % rfecv.n_features_)
# print(rfecv.ranking_)
#
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()

sns.boxplot(x=Data['Discount'])
plt.show()
sns.boxplot(x=Data['CategoryID'])
plt.show()

X=Data[['Discount','DateDif','CategoryID']]
y=Data['OrderPriorityID']

scores = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
print(scores.mean())


knn = KNeighborsClassifier(n_neighbors=595)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores.mean())
#61.57%
# # k_range = list(range(500,600))
# # k_scores = []
# # for k in k_range:
# #      knn = KNeighborsClassifier(n_neighbors=k)
# #      scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
# #      k_scores.append(scores.mean())
# # print(k_scores)
#
# plt.plot(k_range,k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated Accuracy(Order Priority)')
# plt.show()



name=('KNN with K=595','Logistic')
X_axis=np.arange(len(name))
Y_axis=[61.57,62.73]
plt.bar(X_axis,Y_axis,align='center',width=.2)
plt.xticks(X_axis,name)
plt.xlabel('Algorithms')
plt.ylabel('Accuracy %')
plt.title("Performance graph of Classification Algorithms(Order Priority prediction)")
plt.show()