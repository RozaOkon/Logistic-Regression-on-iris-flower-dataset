import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

iris = datasets.load_iris()
X = iris["data"][:, 3].astype(np.float)
y_true = (iris["target"] == 2).astype(np.int)

# Start parameters
a = 3
b = - 7

err_list = []
a_list = []

for j in range(0, len(X)):
    err = 0
    for i in range(0, len(X)):
        y_pred = 1 / (1 + np.exp(-(a * X[i] + b)))
        if y_true[i] == 1:
            err += - np.log(y_pred)
        else:
            err += - np.log(1 - y_pred)
    err_list.append(err)
    a_list.append(a)
    a += 0.1

a_min = a_list[err_list.index(min(err_list))]
print(a_min, a_list.index(a_min))
a = a_min


err_b_list = []
b_list = []

for j in range(0, len(X)):
    err_b = 0
    for i in range(0, len(X)):
        y_pred = 1 / (1 + np.exp(-(a * X[i] + b)))
        if y_true[i] == 1:
            err_b += - np.log(y_pred)
        else:
            err_b += - np.log(1 - y_pred)
    err_b_list.append(err_b)
    b_list.append(b)
    b += 0.1

b_min = b_list[err_b_list.index(min(err_b_list))]
print(b_min, b_list.index(b_min))
b = b_min


y_prob1_list = []
y_prob2_list = []

for i in range (0, len(X)):
    y_prob1 = 1 / (1 + np.exp(-(a * X[i] + b)))
    y_prob2 = 1 - 1 / (1 + np.exp(-(a * X[i] + b)))
    y_prob1_list.append(y_prob1)
    y_prob2_list.append(y_prob2)

plt.plot(X, y_prob1_list, "g-", linewidth = 2, label = "Virginica probability")
plt.plot(X, y_prob2_list, "r-", linewidth = 2, label = "Non-virginica probability")
plt.plot(X, y_true, "yo", linewidth = 2, label = "Measured points")
plt.title("Probability distribution for measured points")
plt.legend()
plt.show()


y_prob1_new_list = []
y_prob2_new_list = []
X_new = np.linspace(0, 3, len(X))
for i in range (0, len(X)):
    y_prob1_new = 1 / (1 + np.exp(-(a * X_new[i] + b)))
    y_prob2_new = 1 - 1 / (1 + np.exp(-(a * X_new[i] + b)))
    y_prob1_new_list.append(y_prob1_new)
    y_prob2_new_list.append(y_prob2_new)

plt.plot(X_new, y_prob1_new_list, "g-", linewidth = 2, label = "Virginica probability")
plt.plot(X_new, y_prob2_new_list, "r-", linewidth = 2, label = "Non-virginica probability")
plt.title("Probability distribution for generated points")
plt.legend()
plt.show()
