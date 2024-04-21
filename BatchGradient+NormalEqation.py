###batch-gradient descent implementation
import numpy as np
import pandas as pd
df = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")

insert = df.insert(0,'x_0',1)
#print(df)

n = df.shape[0]   #rows
d = df.shape[1]   #cols
#print(n,d)

###division of data
X = df.iloc[:, 0:d-2]
y = df.iloc[:, d-2]
#print(X, y)

theta = np.zeros(6)
h_x = 0
###hypothesis function
for i in range(0, d-2):
   h_x += theta[i]*X.iloc[:, i]
#print(h_x)

cost_func = sum(0.5*(h_x - y)**2)
print(cost_func)
alpha = 0.0000001
threshold = 0.0001
a = 100000
while a > threshold:
   for j in range(d-2):
      derivative = sum((h_x - y) * X.iloc[:, j])
      theta[j] = theta[j] - alpha * derivative
   h_x += theta[i] * X.iloc[:, i - 1]
   cost_func_new = sum(0.5 * (h_x - y) ** 2)
   print(cost_func_new)
   a = cost_func - cost_func_new
   if a < threshold:
         break
   cost_func = cost_func_new
print(theta)



###normal equations
# import pandas as pd
# import numpy as np
# df = pd.read_csv("/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv")
# #print(df)
# for i in range(0, 60):
#     X = np.array(df[["age", "BMI", "BP", "blood_sugar", "Gender", "disease_score"]])
#     X_trans = np.transpose(X)
#     y = df.iloc[:, -1]
#
# trans_x = np.dot(X_trans, X)
# #print(trans_x)
#
# inverse = np.linalg.inv(trans_x)
# #print(inverse)
#
# trans_y = np.dot(X_trans, y)
# #print(trans_y)
#
# theta = np.dot(inverse, trans_y)
# print(theta)

