import numpy as np
import itertools
import collections
import sklearn.base

class BagsVectorizer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__( self ):
        pass 
    
    def fit( self, X, y = None ):
        self.feature_names = [test[0] for test in collections.Counter(itertools.chain.from_iterable(X)).most_common()]
        return self
    
    def transform( self, X, y = None ):
        matrix = np.zeros((len(X),len(self.feature_names)))
        for i,bag in enumerate(X):
            for test in bag:
                try:
                    matrix[i,self.feature_names.index(test)] = 1
                except ValueError:
                    pass
        return matrix