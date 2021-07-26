# #############################################################################
# Generate sample data
import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from ml_scratch.knn import KNNRegressor

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
n_neighbors = 10


knn = neighbors.KNeighborsRegressor(n_neighbors)
y_ = knn.fit(X, y).predict(T)

knn_2 = KNNRegressor(n_neighbors)

y_2 = knn_2.fit(X, y).predict(T)

plt.figure(0)
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(T, y_, color='navy', label='prediction')

plt.legend()
plt.title("KNeighborsRegressor")

plt.figure(1)
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(T, y_2, color='green', label='prediction')

plt.legend()
plt.title("Scratch KNNRegressor")

plt.show()

