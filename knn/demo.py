from knn import KNNRegressor
import numpy as np

model = KNNRegressor(3)

A_x = np.array([[1, 2, 3, 4], 
            [5, 5, 6, 7], 
            [1, 4, 2, 3],
            [3, 1, 3, 7],
            [2, 8, 1, 9],
            [1, 5, 9, 7]])

# A_y = ['A', 'A', 'A', 'B', 'B', 'B']
A_y = [1.0, 2.5, 0.6, -1.0, 5.0, 3.0]

B_x = np.array([[5, 3, 2, 1],
            [8, 1, 2, 2]])

model.fit(A_x, A_y)
print(model.predict(B_x))
