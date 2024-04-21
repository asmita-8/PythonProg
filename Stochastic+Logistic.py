import numpy as np
from sklearn.metrics import r2_score
import pandas as pd

df = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")

insert = df.insert(0,'x_0',1)
#print(df)

n = df.shape[0]   #rows
d = df.shape[1]   #cols
#print(n,d)

###division of data
X = df.iloc[4, 0:d-2]
y = df.iloc[4, d-2]
#print(X, y)

theta = np.zeros(6)
h_x = 0
###hypothesis function
h_x += theta*X
#print(h_x)
cost_func = sum(0.5*(h_x - y)**2)
#print(cost_func)
alpha = 0.0000001
threshold = 0.01
a = 100000
derivative = (h_x - y) * X
#print(derivative)
while a > threshold:
      theta = theta - alpha * derivative
      h_x += theta * X
      cost_func_new = sum(0.5 * (h_x - y) ** 2)
      print(cost_func_new)
      a = cost_func - cost_func_new
      if a < threshold:
         break
      cost_func = cost_func_new
print(theta)


###implementation of sigmoid/Logistic function

# import numpy as np
# from matplotlib import pyplot as plt
# z = np.linspace(-50, 50, num=100)
# g_z = 1/(1 + np.exp(-z))
# plt.plot(z, g_z)
# plt.show()


###compute derivative of the sigmoid function
# import numpy as np
# from matplotlib import pyplot as plt
# z = np.linspace(-50, 50, num=100)
# g_z = 1/(1 + np.exp(-z))
# derivative = np.gradient(g_z, z)
# plt.plot(z, g_z, label='g_z')
# plt.plot(z, derivative, label='derivative')
# plt.legend()
# plt.show()


###Implement logistic regression using scikit-learn for the breast cancer dataset and simulated dataset
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from matplotlib import pyplot as plt
# def load_data():
#     [X, y] = load_breast_cancer(return_X_y=True)
#     #print( [X, y])
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, shuffle=False)
#     #print(X_train, X_test, y_train, y_test)
#     logreg = LogisticRegression(max_iter=10000).fit(X_train, y_train)
#     y_pred = logreg.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print(acc)
# def main():
#     load_data()
# if __name__=="__main__":
#     main()
# 

