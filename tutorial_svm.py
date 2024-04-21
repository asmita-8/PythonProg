###TUTORIAL

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import svm
# from sklearn.inspection import DecisionBoundaryDisplay
#
# X = np.array(
#     [
#         [0.4, -0.7],
#         [-1.5, -1.0],
#         [-1.4, -0.9],
#         [-1.3, -1.2],
#         [-1.1, -0.2],
#         [-1.2, -0.4],
#         [-0.5, 1.2],
#         [-1.5, 2.1],
#         [1.0, 1.0],
#         [1.3, 0.8],
#         [1.2, 0.5],
#         [0.2, -2.0],
#         [0.5, -2.4],
#         [0.2, -2.3],
#         [0.0, -2.7],
#         [1.3, 2.1],
#     ]
# )
#
# y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
#
# # Plotting settings
# fig, ax = plt.subplots(figsize=(4, 3))
# x_min, x_max, y_min, y_max = -3, 3, -3, 3
# ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
#
# # Plot samples by color and add legend
# scatter = ax.scatter(X[:, 0], X[:, 1], s=150, c=y, label=y, edgecolors="k")
# ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
# ax.set_title("Samples in two-dimensional feature space")
# _ = plt.show()
#
#
# def plot_training_data_with_decision_boundary(kernel):
#     # Train the SVC
#     clf = svm.SVC(kernel=kernel, gamma=2).fit(X, y)
#
#     # Settings for plotting
#     _, ax = plt.subplots(figsize=(4, 3))
#     x_min, x_max, y_min, y_max = -3, 3, -3, 3
#     ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
#
#     # Plot decision boundary and margins
#     common_params = {"estimator": clf, "X": X, "ax": ax}
#     DecisionBoundaryDisplay.from_estimator(
#         **common_params,
#         response_method="predict",
#         plot_method="pcolormesh",
#         alpha=0.3,
#     )
#     DecisionBoundaryDisplay.from_estimator(
#         **common_params,
#         response_method="decision_function",
#         plot_method="contour",
#         levels=[-1, 0, 1],
#         colors=["k", "k", "k"],
#         linestyles=["--", "-", "--"],
#     )
#
#     # Plot bigger circles around samples that serve as support vectors
#     ax.scatter(
#         clf.support_vectors_[:, 0],
#         clf.support_vectors_[:, 1],
#         s=250,
#         facecolors="none",
#         edgecolors="k",
#     )
#     # Plot samples by color and add legend
#     ax.scatter(X[:, 0], X[:, 1], c=y, s=150, edgecolors="k")
#     ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
#     ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")
#
#     _ = plt.show()
# def main():
#     #kernel = 'linear'
#     #kernel = 'poly'
#     #kernel = 'rbf'
#     kernel = 'sigmoid'
#     plot_training_data_with_decision_boundary(kernel)
# if __name__=="__main__":
#     main()


###Consider the following dataset. Implement the RBF kernel. Check if RBF kernel separates the data well and compare it with Polynomial Kernel.

# def rbf(x1, x2, label):
#      result = []
#      gamma = 2
#      for i in range(len(x1)):
#         diff = x1[i]-x2[i]
#         n = norm(diff)
#         n_sq = n*n
#         mul_gamma = -(gamma*n_sq)
#         exp = np.exp(mul_gamma)
#         result.append(exp)
#      #print(result)
# def plot(x1,x2,label):
#      plt.scatter(x1, x2, c=label)
#      #plt.show()
#      rbf(x1, x2, label)
# def main():
#      x1 = [6, 6, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14]
#      x2 = [5, 9, 6, 8, 10, 2, 5, 10, 13, 5, 8, 6, 11, 4, 8]
#      label = ['Blue', 'Blue', 'Red', 'Red', 'Red', 'Blue', 'Red', 'Red', 'Blue', 'Red', 'Red', 'Red', 'Blue', 'Blue', 'Blue']
#      plot(x1, x2, label)
# if __name__=="__main__":
#      main()


# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import svm
# from sklearn.inspection import DecisionBoundaryDisplay
#
# X = np.array([[6, 5], [6, 9], [8, 6], [8, 8], [8, 10], [9, 2], [9, 5], [10, 10], [10, 13], [11, 5], [11, 8], [12, 6], [12, 11], [13, 4], [14, 8]])
# y = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0])
# # Plotting settings
# fig, ax = plt.subplots(figsize=(4, 3))
# x_min, x_max, y_min, y_max = 4, 16, -2, 16
# ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
#
# # Plot samples by color and add legend
# scatter = ax.scatter(X[:, 0], X[:, 1], s=150, c=y, label=y, edgecolors="k")
# ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
# ax.set_title("Samples in two-dimensional feature space")
# _ = plt.show()
# def plot_training_data_with_decision_boundary(kernel):
#     # Train the SVC
#     clf = svm.SVC(kernel=kernel, gamma=2).fit(X, y)
#
#     # Settings for plotting
#     _, ax = plt.subplots(figsize=(4, 3))
#     x_min, x_max, y_min, y_max = 4, 16, -2, 16
#     ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
#
#     # Plot decision boundary and margins
#     common_params = {"estimator": clf, "X": X, "ax": ax}
#     DecisionBoundaryDisplay.from_estimator(
#         **common_params,
#         response_method="predict",
#         plot_method="pcolormesh",
#         alpha=0.3,
#     )
#     DecisionBoundaryDisplay.from_estimator(
#         **common_params,
#         response_method="decision_function",
#         plot_method="contour",
#         levels=[-1, 0, 1],
#         colors=["k", "k", "k"],
#         linestyles=["--", "-", "--"],
#     )
#
#     # Plot bigger circles around samples that serve as support vectors
#     ax.scatter(
#         clf.support_vectors_[:, 0],
#         clf.support_vectors_[:, 1],
#         s=250,
#         facecolors="none",
#         edgecolors="k",
#     )
#     # Plot samples by color and add legend
#     ax.scatter(X[:, 0], X[:, 1], c=y, s=150, edgecolors="k")
#     ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
#     ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")
#     _ = plt.show()
# def main():
#     kernel = 'poly'
#     #kernel = 'rbf'
#     plot_training_data_with_decision_boundary(kernel)
# if __name__=="__main__":
#     main()


###Try classifying classes 1 and 2 from the iris dataset with SVMs, with the 2 first features. Leave out 10% of each class and test prediction
###performance on these observations.

# data = pd.read_csv("/home/ibab/movedfiles/Iris.csv")
# df = data.iloc[26:65, :]
# #print(df)
# features = df.iloc[:, 1:3]
# label = df.iloc[:, -1]
# X = np.array(features)
# y = np.array(label)
# app = []
# for i in range(len(y)):
#     if y[i] == 'Iris-setosa':
#         app.append(0)
#     else:
#         app.append(1)
# y = np.array(app)
# #print(x, y)
# # Plotting settings
# fig, ax = plt.subplots(figsize=(4, 3))
# x_min, x_max, y_min, y_max = 4, 8, -2, 8
# ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
#
# # Plot samples by color and add legend
# scatter = ax.scatter(X[:, 0], X[:, 1], s=150, c=y, label=y, edgecolors="k")
# ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
# ax.set_title("Samples in two-dimensional feature space")
# _ = plt.show()
# def plot_training_data_with_decision_boundary(kernel):
#     # Train the SVC
#     clf = svm.SVC(kernel=kernel, gamma=2).fit(X, y)
#
#     # Settings for plotting
#     _, ax = plt.subplots(figsize=(4, 3))
#     x_min, x_max, y_min, y_max = 4, 8, -2, 8
#     ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
#
#     # Plot decision boundary and margins
#     common_params = {"estimator": clf, "X": X, "ax": ax}
#     DecisionBoundaryDisplay.from_estimator(
#         **common_params,
#         response_method="predict",
#         plot_method="pcolormesh",
#         alpha=0.3,
#     )
#     DecisionBoundaryDisplay.from_estimator(
#         **common_params,
#         response_method="decision_function",
#         plot_method="contour",
#         levels=[-1, 0, 1],
#         colors=["k", "k", "k"],
#         linestyles=["--", "-", "--"],
#     )
#
#     # Plot bigger circles around samples that serve as support vectors
#     ax.scatter(
#         clf.support_vectors_[:, 0],
#         clf.support_vectors_[:, 1],
#         s=250,
#         facecolors="none",
#         edgecolors="k",
#     )
#     # Plot samples by color and add legend
#     ax.scatter(X[:, 0], X[:, 1], c=y, s=150, edgecolors="k")
#     ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
#     ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")
#     _ = plt.show()
# def main():
#     #kernel = 'poly'
#     kernel = 'rbf'
#     plot_training_data_with_decision_boundary(kernel)
# if __name__=="__main__":
#     main()

