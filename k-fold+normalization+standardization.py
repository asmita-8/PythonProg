##K-fold cross validation. Implement for K = 10. Implement from scratch, then, use scikit-learn methods.

# import pandas as pd
# import numpy as np
# import math
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# df = pd.read_csv("/home/ibab/movedfiles/binary_logistic_regression_data_simulated_for_ML.csv")
# #print(df)
# n = df.shape[0]
# d = df.shape[1]
# k = 10
# fold = math.floor(n/k)
# accuracy_list = []
# def k_fold():
#     for i in range(0, k):
#         x_test = df.iloc[i*fold:(i+1)*fold, 0:d-1]
#         y_test = df.iloc[i*fold:(i+1)*fold, d-1]
#         x_train = pd.concat([df.iloc[0:i*fold, 0:d-1], df.iloc[(i+1)*fold:n, 0:d-1]])
#         y_train = pd.concat([df.iloc[0:i*fold, d-1], df.iloc[(i+1)*fold:n, d-1]])
#         model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
#         y_pred = model.predict(x_test)
#         acc = accuracy_score(y_test, y_pred)
#         accuracy_list.append(acc)
#     mean = np.mean(accuracy_list)
#     std_dev = np.std(accuracy_list)
#     print(accuracy_list, "\n", mean, "\n", std_dev)
# def main():
#     k_fold()
# if __name__=="__main__":
#     main()


###Perform 10-fold cross validation for SONAR dataset in scikit-learn using logistic regression. SONAR dataset is a
###binary classification problem with target variables as Metal or Rock. i.e. signals are from metal or rock.
# import pandas as pd
# from sklearn.model_selection import KFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import numpy as np
# df = pd.read_csv("/home/ibab/movedfiles/Sonar.csv")
# #print(df)
# n = df.shape[0]
# d = df.shape[1]
# x = df.iloc[:, 0:d-1]
# y = df.iloc[:, d-1]
# k = 10
# #print(x,y)
# kf = KFold(n_splits=k)
# model = LogisticRegression()
# acc_score = []
# for train_ind, test_ind in kf.split(x):
#     x_train, x_test = x.iloc[train_ind, :], x.iloc[test_ind, :]
#     y_train, y_test = y[train_ind], y[test_ind]
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     acc = accuracy_score(y_pred, y_test)
#     acc_score.append(acc)
# mean = np.mean(acc_score)
# std = np.std(acc_score)
# print(acc_score, "\n", mean, "\n", std)


##Data normalization - scale the values between 0 and 1. Implement code from scratch.
# import pandas as pd
# df = pd.read_csv("/home/ibab/movedfiles/Sonar.csv")
# #print(df)
# df_new1 = pd.DataFrame(df)
# n = df.shape[0]
# d = df.shape[1]
# for j in range(d-1):
#     mini = df.iloc[:, j].min()
#     maxi = df.iloc[:, j].max()
#     df_new1.iloc[:, j] = (df.iloc[:, j]-mini)/maxi-mini
# print(df_new1)



##Data standardization - scale the values such that mean of new dist = 0 and sd = 1. Implement code from scratch.
import pandas as pd
import numpy as np
df = pd.read_csv("/home/ibab/movedfiles/Sonar.csv")
#df_new = pd.DataFrame(df)
df_new = df.copy()
n = df.shape[0]
d = df.shape[1]

for j in range(d-1):
     m = np.mean(df.iloc[:, j])
     st_d = np.std(df.iloc[:, j])
     df_new.iloc[:, j] = (df.iloc[:, j]-m)/st_d
m_new = np.mean(df_new, axis=0)
st_d_new = np.std(df_new, axis=0)
print(df_new)
print(m_new)
print(st_d_new)

##k-fold after normalization
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
df = pd.read_csv("/home/ibab/movedfiles/Sonar.csv")
#print(df)
df_new1 = df.copy()
n = df.shape[0]
d = df.shape[1]
for j in range(d-1):
    mini = df.iloc[:, j].min()
    maxi = df.iloc[:, j].max()
    df_new1.iloc[:, j] = (df.iloc[:, j]-mini)/maxi-mini
x = df_new1.iloc[:, 0:d-1]
y = df_new1.iloc[:, d-1]
k = 10
#print(x,y)
kf = KFold(n_splits=k)
model = LogisticRegression()
acc_score = []
for train_ind, test_ind in kf.split(x):
    x_train, x_test = x.iloc[train_ind, :], x.iloc[test_ind, :]
    y_train, y_test = y[train_ind], y[test_ind]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_pred, y_test)
    acc_score.append(acc)
mean = np.mean(acc_score)
std = np.std(acc_score)
print(acc_score, "\n", mean, "\n", std)


##k-fold after standardization
# import pandas as pd
# from sklearn.model_selection import KFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import numpy as np
# df = pd.read_csv("/home/ibab/movedfiles/Sonar.csv")
# #print(df)
# df_new = df.copy()
# n = df.shape[0]
# d = df.shape[1]
# for j in range(d-1):
#     m = np.mean(df.iloc[:, j])
#     st_d = np.std(df.iloc[:, j])
#     df_new.iloc[:, j] = (df.iloc[:, j]-m)/st_d
# x = df.iloc[:, 0:d-1]
# y = df.iloc[:, d-1]
# k = 10
# #print(x,y)
# kf = KFold(n_splits=k)
# model = LogisticRegression()
# acc_score = []
# for train_ind, test_ind in kf.split(x):
#     x_train, x_test = x.iloc[train_ind, :], x.iloc[test_ind, :]
#     y_train, y_test = y[train_ind], y[test_ind]
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     acc = accuracy_score(y_pred, y_test)
#     acc_score.append(acc)
# mean = np.mean(acc_score)
# std = np.std(acc_score)
# print(acc_score, "\n", mean, "\n", std)

