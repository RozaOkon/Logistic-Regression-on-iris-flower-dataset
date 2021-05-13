import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

X = iris["data"]
y = iris["target"]

log_reg = LogisticRegression(random_state=42)
clf = log_reg.fit(X, y)
X_new = np.linspace(0, 8, 1000).reshape(-1, 4)
y_proba_new = log_reg.predict_proba(X_new)
y_proba = log_reg.predict_proba(X)

plt.clf()
plt.plot(X_new, y_proba_new[:, 0], "g-", linewidth=2, label="Setosa")
plt.plot(X_new, y_proba_new[:, 1], "r-", linewidth=2, label="Versicolor")
plt.plot(X_new, y_proba_new[:, 2], "y-", linewidth=2, label="Virginica")
plt.legend()
plt.title('Probability distribution for genetared points')
plt.show()

plt.clf()
plt.plot(X, y_proba[:, 0], "g-", linewidth=2, label="Setosa")
plt.plot(X, y_proba[:, 1], "r-", linewidth=2, label="Versicolor")
plt.plot(X, y_proba[:, 2], "y-", linewidth=2, label="Virginica")
plt.legend()
plt.title('Probability distribution for measured points')
plt.show()
