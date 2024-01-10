###ques:1
# import numpy as np
# m = np.array([1, 2, 3, 4, 5, 6])
# a = m.reshape(2, 3)
# a_trans = np.transpose(a)
# result = np.dot(a_trans, a)
# print(result)


###ques:2
# import numpy as np
# from matplotlib import pyplot as plt
# x = np.linspace(-100, 100, num=100)
# y = 2*x + 3
# plt.plot(x, y)
# plt.show()


###ques:3
# import numpy as np
# from matplotlib import pyplot as plt
# x = np.linspace(-10, 10, num=100)
# y = 4 + 3*x + 2*x*x
# plt.plot(x, y)
# plt.show()


###ques:4
# import numpy as np
# from matplotlib import pyplot as plt
# mean = 10
# sigma = 15
# sqrt = np.sqrt(2*3.14*(sigma*sigma))
# x = np.linspace(-100, 100, num=100)
# numerator = (x-mean)*(x-mean)
# denominator = 2*(sigma*sigma)
# y = 1/sqrt * np.exp(-(numerator/denominator))
# plt.plot(x, y)
# plt.show()


###ques:5
# import numpy as np
# from matplotlib import pyplot as plt
# x = np.linspace(-100, 100, num=100)
# y = x*x
# d = np.gradient(y, x)
# plt.plot(x, y, label='f(x)')
# plt.plot(x, d, label='dy/dx')
# plt.legend()
# plt.show()

###simulated data for ML
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
# def load():
#     df = pd.read_csv(r"/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")
#     #print(df)
#     X = df.iloc[:, 0:-2]
#     y = df.iloc[:, -2]
#     #print(X)
#     X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.3, train_size=0.7)
#     #print(X_train, X_test, y_train,y_test)
#     model=LinearRegression().fit(X_train, y_train)
#     y_pred =model.predict(X_test)
#     r2 = r2_score(y_test, y_pred)
#     print(r2)
# def main():
#     load()
# if __name__=="__main__":
#     main()
