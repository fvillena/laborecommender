import numpy as np
import itertools
import collections
import sklearn.base

class BagsVectorizer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Convert a collection of bags to a binary matrix of tests counts.

    This produces a matrix representation of the presence of a test on a given
    bag of laboratory tests.

    Attributes
    ----------
    feature_names : list of str
        A list of laboratory tests present in the collection.
    
    Examples
    --------
    >>> import laborecommender.features
    >>> bags = [["a","b","c"],["d","e","f"]]
    >>> vectorizer = laborecommender.features.BagsVectorizer()
    >>> vectorizer.fit(bags)
    >>> vectorizer.transform(bags)
    array([ [1., 1., 1., 0., 0., 0.],
            [0., 0., 0., 1., 1., 1.]])

    """
    def __init__( self ):
        pass 
    
    def fit( self, X: list, y = None ):
        """
        Learn a set of laboratory test names.

        Parameters
        ----------
        X : list of list of str
            A list of laboratory test bags.
        
        Returns
        -------
        self

        """
        self.feature_names = [test[0] for test in collections.Counter(itertools.chain.from_iterable(X)).most_common()]
        return self
    
    def transform( self, X, y = None ):
        """
        Transform a list of laboratory test bags to a bag-test matrix.

        Extract the presence of a given laboratory test from bags using the set of
        laboratory tests fitted.

        Parameters
        ----------
        X : list of list of str
            A list of laboratory test bags.
        
        Returns
        -------
        matrix of shape (`len(X)`, `len(feature_names)`)
            Bag-test matrix.
            
        """
        matrix = np.zeros((len(X),len(self.feature_names)))
        for i,bag in enumerate(X):
            for test in bag:
                try:
                    matrix[i,self.feature_names.index(test)] = 1
                except ValueError:
                    pass
        return matrix