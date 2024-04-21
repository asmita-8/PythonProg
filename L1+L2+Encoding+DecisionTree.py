##Implement L2-norm and L1-norm
import pandas as pd
##L2
from sklearn.datasets import fetch_california_housing
# import numpy as np
# import pandas as pd
# df = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")
# insert = df.insert(0,'x_0',1)
# #print(df)
#
# n = df.shape[0]   #rows
# d = df.shape[1]   #cols
# #print(n,d)
#
# ###division of data
# X = df.iloc[:, 0:d-2]
# y = df.iloc[:, d-2]
# #print(X, y)
#
# theta = np.zeros(6)
# h_x = 0
# ###hypothesis function
# for i in range(0, d-2):
#    h_x += theta[i]*X.iloc[:, i]
# #print(h_x)
#
# cost_func = sum(0.5*(h_x - y)**2)+(0.01*sum((theta)**2))
# print(cost_func)
# alpha = 0.0000001
# threshold = 0.0001
# a = 100000
# while a > threshold:
#    for j in range(d-2):
#       derivative = sum((h_x - y) * X.iloc[:, j])+2*0.01*theta[j]
#       theta[j] = theta[j] - alpha * derivative
#    h_x += theta[i] * X.iloc[:, i - 1]
#    cost_func_new = sum(0.5 * (h_x - y) ** 2)
#    print(cost_func_new)
#    a = cost_func - cost_func_new
#    if a < threshold:
#          break
#    cost_func = cost_func_new
# print(theta)


##L1

# import numpy as np
# import pandas as pd
# df = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")
#
# insert = df.insert(0,'x_0',1)
# #print(df)
#
# n = df.shape[0]   #rows
# d = df.shape[1]   #cols
# #print(n,d)
#
# ###division of data
# X = df.iloc[:, 0:d-2]
# y = df.iloc[:, d-2]
# #print(X, y)
#
# theta = np.zeros(6)
# h_x = 0
# ###hypothesis function
# for i in range(0, d-2):
#    h_x += theta[i]*X.iloc[:, i]
# #print(h_x)
# penalty = 0.01*sum(abs(theta))
# cost_func = sum(0.5*(h_x - y)**2)+penalty
# print(cost_func)
# alpha = 0.0000001
# threshold = 0.0001
# a = 100000
# while a > threshold:
#    for j in range(d-2):
#       if penalty > 0:
#          derivative = sum((h_x - y) * X.iloc[:, j]) + 1
#       else:
#          derivative = sum((h_x - y) * X.iloc[:, j]) - 1
#       theta[j] = theta[j] - alpha * derivative
#    h_x += theta[i] * X.iloc[:, i - 1]
#    cost_func_new = sum(0.5 * (h_x - y) ** 2)
#    print(cost_func_new)
#    a = cost_func - cost_func_new
#    if a < threshold:
#          break
#    cost_func = cost_func_new
# print(theta)

##Build a classification model for wisconsin dataset using Ridge and Lasso classifier using scikit-learn

##ridge

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
[X, y] = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7)
model = RidgeClassifier(alpha=0.01)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(score)

##lasso

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score
[X, y] = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7)
model = LogisticRegression( max_iter=1000 ,penalty='l1', solver='liblinear')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(score)

#Use breast_cancer.csv (https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv)and use scikit learn
# methods, OrdinalEncoder, OneHotEncoder(sparse=False), LabelEncoder to implement complete Logistic Regression Model.

#ORDINAL

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.preprocessing import LabelEncoder
# df = pd.read_csv("/home/ibab/Downloads/b_cancer.csv")
# #print(df)
# data = df.values
# x = data[:, 0:-1].astype(str)
# y = data[:, -1].astype(str)
# #print(x, y)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
# o_encoder = OrdinalEncoder()
# o_encoder.fit_transform(x_train)
# x_train = o_encoder.transform(x_train)
# x_test = o_encoder.transform(x_test)
# l_encoder = LabelEncoder()
# l_encoder.fit_transform(y_train)
# y_train = l_encoder.transform(y_train)
# y_test = l_encoder.transform(y_test)
# model = LogisticRegression(max_iter=10000).fit(x_train, y_train)
# pred = model.predict(x_test)
# acc = accuracy_score(y_test, pred)
# print("Accuracy is :", acc)


#ONE HOT ENCODING

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import LabelEncoder
# from sklearn. metrics import accuracy_score
# df = pd.read_csv("/home/ibab/Downloads/b_cancer.csv")
# # print(df)
# data = df.values
# x = data[:, 0:-1].astype(str)
# y = data[:, -1].astype(str)
# #print(x, y)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
# o_encoder = OneHotEncoder()
# o_encoder.fit_transform(x_train)
# x_train = o_encoder.transform(x_train)
# x_test = o_encoder.transform(x_test)
# l_encoder = LabelEncoder()
# l_encoder.fit_transform(y_train)
# y_train = l_encoder.transform(y_train)
# y_test = l_encoder.transform(y_test)
# model = LogisticRegression(max_iter=10000).fit(x_train, y_train)
# pred = model.predict(x_test)
# acc = accuracy_score(y_test, pred)
# print("Accuracy is :", acc)

#Write a program to partition a dataset (simulated data for regression)  into two parts, based on a feature (BP) and
# for a threshold, t = 80. Generate additional two partitioned datasets based on different threshold values of t = [78, 82].
# import pandas as pd
# df = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")
# print(df)
#
# threshold = 80
# df_more = df[df["BP"] > threshold].copy()
# df_less = df[df["BP"] <= threshold].copy()
# print(df_more)
# print(df_less)
#
# threshold = 78
# df_more = df[df["BP"] > threshold].copy()
# df_less = df[df["BP"] <= threshold].copy()
# print(df_more)
# print(df_less)
#
# threshold = 82
# df_more = df[df["BP"] > threshold].copy()
# df_less = df[df["BP"] <= threshold].copy()
# print(df_more)
# print(df_less)


##Implement a regression decision tree algorithm using scikit-learn for the simulated dataset.

# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.tree import DecisionTreeRegressor
# from sklearn import tree
# df = pd.read_csv("/home/ibab/movedfiles/simulated_data_multiple_linear_regression_for_ML.csv")
# #print(df)
# n = df.shape[0]
# d = df.shape[1]
# x = df.iloc[:, 0:d-1]
# y = df.iloc[:, d-1]
# regressor = DecisionTreeRegressor(random_state=0)
# regressor.fit(x, y)
# tree.plot_tree(regressor, feature_names=df.columns[:-1], filled=True)
# plt.show()


#Implement a classification decision tree algorithm using scikit-learn for the simulated dataset.
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
df = pd.read_csv("/home/ibab/movedfiles/binary_logistic_regression_data_simulated_for_ML.csv")
n = df.shape[0]
d = df.shape[1]
x = df.iloc[:, 0:d-1]
y = df.iloc[:, d-1]
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(x, y)
tree.plot_tree(classifier, feature_names=df.columns[:-1], filled=True)
plt.show()


