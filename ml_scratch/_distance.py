import numpy as np

def manhattan_distance(X_1, X_2):
    """
    Calculate distance using manhattan.

    .. math::
    
        {\sum _{i=1}^{n}  \left| q_{i}-p_{i}\\right| }

    """
    metric = [np.sum(np.abs(X_1 - X), axis=1) for X in X_2] 
    return metric

def euclidean_distance(X_1, X_2):
    """
    Calculate distance using euclidean.

    .. math::

        \sqrt {\sum _{i=1}^{n}  \left( q_{i}-p_{i}\\right)^2 }

    """
    # Distance is not square rooted because:
    # 1. It doesn't really change the amount of distance between each point
    # 2. Slows down calculation
    metric = [np.sum((X_1 - X) ** 2, axis=1) for X in X_2]
    return metric

def cosine_distance(X_1, X_2):
    """
    Calculate distance using cosine.

    .. math::

        {\mathbf {A} \cdot \mathbf {B}  \over \|\mathbf {A} \|\|\mathbf {B} \|}

    """
    metric = [np.dot(X_1, X) / (np.linalg.norm(X_1) * np.linalg.norm(X)) for X in X_2]
    return metric