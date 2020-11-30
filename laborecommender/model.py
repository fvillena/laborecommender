import sklearn.pipeline
import sklearn.neighbors
import sklearn.base
import numpy as np
from . import features
import collections
import itertools

def list_of_bags_to_set(bags: list) -> list:
    """
    Finds the list of different tests from a list of laboratory test bags.

    Parameters
    ----------
    bags : list of list of str
        A list of laboratory test bags.
    
    Returns
    -------
    list of str
        List of the different laboratory test available.

    """
    return [item[0] for item in collections.Counter(itertools.chain.from_iterable(bags)).most_common()]
def remove_items_from_bag(bag: list,banned_items: list):
    """
    Removes given tests from a list of tests.

    Parameters
    ----------
    bag : list of str
        List of laboratory tests
    banned_items : list of str
        Lis of laboratory tests to remove
    
    Returns
    -------
    list of str
        List of laboratory tests.

    """
    return [item for item in bag if item not in banned_items]

class NearestBags(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """
    Unsupervised learner for implementing neighbor bags searches.

    Perform neighbor searches within a list of laboratory test bags
    this class wraps `sklearn.neighbors.NearestNeighbors` class.

    Parameters
    ----------
    k : int, default=10
        Number of neighbors to use by default for queries.
    metric : str, default='jaccard'
        the distance metric to use for the tree.

    Examples
    --------
    >>> import laborecommender.data
    >>> import laborecommender.model
    >>> import laborecommender.features
    >>> bags = laborecommender.data.get_bags_from_mimic()
    >>> vectorizer = laborecommender.features.BagsVectorizer()
    >>> bags_vectorized = vectorizer.fit_transform(bags_vectorized)
    >>> nb = laborecommender.model.NearestBags()
    >>> nb.fit(bags_vectorized)
    >>> nb.predict(bags_vectorized[0].reshape(1, -1))
    array([[ 796,    4,    0,    3,  396, 1595,    1,   94,  195,    2]])

    """
    def __init__(self, k = 10, metric="jaccard"):
        self.k = k
        self.metric = metric

    def fit(self, X, y = None):
        """
        Fit the model using X as training data

        Parameters
        ----------
        X : array-like
            Training bag-test matrix.
        
        Returns
        -------
        self
        
        """
        self.nn = sklearn.neighbors.NearestNeighbors(metric=self.metric)
        self.nn.fit(X)
        return self

    def predict(self, X):
        """
        Finds the K-neighbors of a vectorized bag.

        Returns indices of the neighbors of each bag.

        Parameters
        ----------
        X : array-like
            Bags as a bag-test matrix representation.
        
        Returns
        -------
        array, shape (n_queries, k)
            Indices of the nearest points in the training matrix.

        """
        return self.nn.kneighbors(X, self.k, return_distance=False)

class LaboRecommender():
    """
    Recommends a set of laboratory tests based on already selected tests.

    Based on a list of laboratory tests it recommends the most likely test
    to select next. This class wraps `laborecommender.features.BagsVectorizer` and
    `laborecommender.model.NearestBags`.

    Parameters
    ----------
    k : int, default=10
        Number of neighbors to use by default for queries.
    metric : str, default='jaccard'
        the distance metric to use for the tree.
    
    Attributes
    ----------
    tests_ : list of str
        List of the set of different tests available in the training dataset.
    bags_ : list of list of str
        List of laboratory tests bags in the training dataset.
    
    Examples
    --------
    >>> import laborecommender.model
    >>> import laborecommender.data
    >>> bags = laborecommender.data.get_bags_from_mimic()
    >>> lr = laborecommender.model.LaboRecommender()
    >>> lr.fit(bags)
    >>> lr.predict([bags[0][:3]])
    [['Chloride', 'Potassium', 'Anion Gap', 'Creatinine', 'Urea Nitrogen']]

    """
    def __init__(self,k = 10, metric="jaccard"):
        self.k = k
        self.metric = metric
    def set_params(self,k,metric):
        """
        Set the parameters of this estimator

        Parameters
        ----------
        k : int, default=10
        Number of neighbors to use by default for queries.
        metric : str, default='jaccard'
            the distance metric to use for the tree.
        
        Returns
        -------
        self

        """
        self.k = k
        self.metric = metric
        return self
    def fit(self, bags):
        """
        Computes a bag-test matrix and trains a nearest bags estimator.

        This class computes a bag-test matrix using laborecommender.features.BagsVectorizer
        and trains a nearest neighbors searcher using laborecommender.model.NearestBags.

        Parameters
        ----------
        bags : list of list of str
            A list of laboratory test bags.
        
        Returns
        -------
        self

        """
        self.pipe = sklearn.pipeline.Pipeline([
            ("transformer", features.BagsVectorizer()),
            ("n", NearestBags(self.k,self.metric))
        ])
        self.pipe.fit(bags)
        self.bags_ = np.array(bags)
        self.tests_ = self.pipe["transformer"].feature_names # pylint: disable=no-member
        return self
    def predict(self, bags, n=5):
        """
        Finds the most likely to select tests.

        From an already selected list of laboratory tests finds the `n most likely
        to select laboratory tests 

        Parameters
        ----------
        bags : list of list of str
            A list of laboratory test bags.
        n : int
            Number of tests to return.
        
        Returns
        list of list str
            List of lists of most likely to select laboratory tests.

        """
        self.n=n
        recommended_bag_ids = self.pipe.predict(bags)
        recommended_bags = self.bags_[recommended_bag_ids]
        recommendations = map(list_of_bags_to_set,recommended_bags)
        recommendations = [remove_items_from_bag(recommendation,bag)[:self.n] for recommendation,bag in zip(recommendations,bags)]
        return recommendations