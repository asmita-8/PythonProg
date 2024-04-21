##Implement entropy measure using Python. The function should accept a bunch of data points and their class labels
##and return the entropy value.
import pandas as pd
import math
df = pd.read_csv("/home/ibab/movedfiles/Sonar.csv")
n = df.shape[0]
#print(df)
df_positive = df[df["Class"] == 1].copy()
n_p = df_positive.shape[0]
df_negative = df[df["Class"] == 0].copy()
n_n = df_negative.shape[0]
p_pos = n_p/n
p_neg = n_n/n
entropy = -((p_pos*math.log(p_pos, 2)) + (p_neg*math.log(p_neg, 2)))
print(entropy)

