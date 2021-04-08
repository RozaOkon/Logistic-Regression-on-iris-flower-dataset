import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

X = iris["data"][:, 3:]
y = iris["target"]

plt.hist(X[y==0, :], color = "darkred")
plt.hist(X[y==1, :], color = "firebrick")
plt.hist(X[y==2, :], color = "lightcoral")
plt.show()

log_reg = LogisticRegression(random_state=42)
clf = log_reg.fit(X, y)
X_new = np.linspace(0, 2.5, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X)
y_proba_new = log_reg.predict_proba(X_new)
print(y_proba)
print(y_proba_new)

plt.clf()
plt.plot(X_new, y_proba_new[:, 0], "g-", linewidth=2, label="Setosa")
plt.plot(X_new, y_proba_new[:, 1], "r-", linewidth=2, label="Versicolor")
plt.plot(X_new, y_proba_new[:, 2], "y-", linewidth=2, label="Virginica")
plt.legend()
plt.show()

plt.plot(X, y_proba[:, 0], "g-", linewidth=2, label="Setosa")
plt.plot(X, y_proba[:, 1], "r-", linewidth=2, label="Versicolor")
plt.plot(X, y_proba[:, 2], "y-", linewidth=2, label="Virginica")
plt.legend()
plt.show()

