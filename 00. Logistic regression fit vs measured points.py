import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
print(list(iris.keys()))

# print(iris.data.shape)
# print(len(iris.data))
# print(iris.target.shape)

X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)
print(iris["target"])

plt.hist(X[y==0, :], color = "green")
plt.hist(X[y==1, :], color = "red")
plt.xlim(0,3)
plt.show()

log_reg = LogisticRegression(random_state=42)
clf = log_reg.fit(X, y)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

plt.clf()
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Virginica probability")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Non-virginica probability")
plt.title("Probability distribution for generated points - sklearn library")
plt.legend()
plt.show()

y_proba_new = log_reg.predict_proba(X)
plt.clf()
plt.plot(X, y_proba_new[:, 1], "g-", linewidth=2, label="Virginica probability")
plt.plot(X, y_proba_new[:, 0], "b--", linewidth=2, label="Non-virginica probability")
plt.plot(X, y, "yo", linewidth = 2, label = "Measured points")
plt.title("Probability distribution for measured points - sklearn library")
plt.legend()
plt.show()