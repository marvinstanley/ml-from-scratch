# Principal Component Analysis
from numpy import array
from pca import PCA

###### USE THIS DATA ########
# define a matrix
# A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
A = array([[1, 2, 3, 4], 
            [5, 5, 6, 7], 
            [1, 4, 2, 3], 
            [5, 3, 2, 1],
            [8, 1, 2, 2]])
print(A)
###### USE THIS DATA ########

# create the PCA instance
pca = PCA(2)
# fit on data
pca.fit(A)
# access values and vectors
print(pca.components_)
print(pca.explained_variance_)
# transform data
B = pca.transform(A)
print(B)