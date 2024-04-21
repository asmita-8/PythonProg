###For the heart.csv dataset, build a logistic regression classifier to predict the risk of heart disease.
###Vary the threshold to generate multiple confusion matrices. Implement a python code to calculate the following metrics
###Accuracy
###Precision
###Sensitivity
###Specificity
###F1-score
###Plot the ROC curve
###AUC
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, roc_auc_score
# df = pd.read_csv("/home/ibab/movedfiles/heart.csv")
# #print(df)
# x = df.iloc[:, 0:-1]
# y = df.iloc[:, -1]
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
# model = LogisticRegression(max_iter=10000).fit(x_train, y_train)
# pred = model.predict(x_test)
# # print(pred)
# pred = np.array(pred)
# y_test = np.array(y_test)
# tp = 0
# tn = 0
# fn = 0
# fp = 0
# for i, j in zip(y_test, pred):
#     if i == 1 and j == 1:
#         tp = tp+1
#     if i == 0 and j == 0:
#         tn = tn+1
#     if i == 1 and j == 0:
#         fn = fn+1
#     if i == 0 and j == 1:
#         fp = fp+1
# print(tp, fp, fn, tn)
# total = tp+fp+fn+tn
# accuracy = (tp+tn)/total if total != 0 else 0
# precision = tp/(tp+fp) if (tp+fp) != 0 else 0
# sensitivity = tp/(tp+fn) if (tp+fn) != 0 else 0
# specificity = tn/(tn+fp) if (tn+fp) != 0 else 0
# f1 = 2*((precision*sensitivity)/(precision+sensitivity)) if (precision+sensitivity) != 0 else 0
# print(accuracy, '\n', precision, '\n', sensitivity, '\n', specificity, '\n', f1)
# sensitivity, specificity, _ = roc_curve(y_test, pred)
# auc = roc_auc_score(y_test, pred)
# plt.plot(sensitivity, specificity, label="AUC=" + str(auc))
# plt.ylabel('Specificity')
# plt.xlabel('Sensitivity')
# plt.legend(loc=2)
# plt.show()
