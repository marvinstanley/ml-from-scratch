# Principal Component Analysis
from numpy import array
from sklearn.decomposition import PCA
from ml_scratch.pca import PCA as PCA_Scratch

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
pca = PCA(4)
# fit on data
pca.fit(A)
# access values and vectors
print(pca.components_)
print(pca.explained_variance_)
# transform data
B = pca.transform(A)
print(B)

# create the PCA instance
pca_2 = PCA_Scratch(4, method='svd')
# fit on data
pca_2.fit(A)
# access values and vectors
print(pca_2.components_)
print(pca_2.explained_variance_)
# transform data
B = pca_2.transform(A)
print(B)