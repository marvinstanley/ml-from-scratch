# Principal Component Analysis
from numpy import array
from sklearn.decomposition import PCA

###### USE THIS DATA ########
# define a matrix
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
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