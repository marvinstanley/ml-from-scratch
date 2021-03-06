# #############################################################################
# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
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


knn = KNNRegressor(n_neighbors)
y_ = knn.fit(X, y).predict(T)

plt.scatter(X, y, color='darkorange', label='data')
plt.plot(T, y_, color='navy', label='prediction')

plt.legend()
plt.title("KNeighborsRegressor")

plt.show()

