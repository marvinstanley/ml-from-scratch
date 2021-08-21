# ml-from-scratch
Machine Learning methods implementation from scratch

Created for PACMANN AI Research Scientist Recruitment test

Currently have implemented PCA and KNNRegressor + KNNClassifier

See https://marvinstanley.github.io/ml-from-scratch/ for the docs

## How to Install
You can install this module by

1. Clone this repo
    `git clone https://github.com/marvinstanley/ml-from-scratch.git`
2. Change directory
    `cd ml-from-scratch`
3. Install using pip
    `pip install .`

Or you can directly use it by cloning this repo to the root of your project

## How to use
**PCA**
```python
from ml_scratch.pca import PCA
import numpy as np

###### USE THIS DATA ########
# define a matrix
A = np.array([[1, 2, 3, 4], 
            [5, 5, 6, 7], 
            [1, 4, 2, 3], 
            [5, 3, 2, 1],
            [8, 1, 2, 2]])
            
###### USE THIS DATA ########

# # create the PCA instance
pca = PCA(2, method='svd')
# fit on data
pca.fit(A)
# access values and vectors
print(pca.components_)
print(pca.explained_variance_)
# transform data
B = pca.transform(A)
print(B)
```

**KNNRegressor**
```python
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
```
