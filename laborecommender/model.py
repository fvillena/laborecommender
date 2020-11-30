import sklearn.pipeline
import sklearn.neighbors
import sklearn.base
import numpy as np
from . import features
import collections
import itertools

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
    >>> bags = laborecommender.data.get_bags_from_mimic()
    >>> nb = laborecommender.model.NearestBags()
    >>> nb.fit(bags)
    >>> nb.predict(bags[0])

    """
    def __init__(self, k = 10, metric="jaccard"):
        self.k = k
        self.metric = metric
    def fit(self, X, y = None):
        self.nn = sklearn.neighbors.NearestNeighbors(metric=self.metric)
        self.nn.fit(X)
        return self
    def predict(self, X):
        return self.nn.kneighbors(X, self.k, return_distance=False)

class LaboRecommender():
    def __init__(self,k = 10, metric="jaccard"):
        self.k = k
        self.metric = metric
    def set_params(self,k,metric):
        self.k = k
        self.metric = metric
        return self
    def fit(self, bags):
        self.pipe = sklearn.pipeline.Pipeline([
            ("transformer", features.BagsVectorizer()),
            ("n", NearestBags(self.k,self.metric))
        ])
        self.pipe.fit(bags)
        self.bags_ = np.array(bags)
        self.tests_ = self.pipe["transformer"].feature_names # pylint: disable=no-member
        return self
    def predict(self, bags, n=5):
        self.n=n
        recommended_bag_ids = self.pipe.predict(bags)
        recommendations = []
        for i,row in enumerate(recommended_bag_ids):
            row_bags = self.bags_[row]
            recommendation = [item[0] for item in collections.Counter(itertools.chain.from_iterable(row_bags)).most_common()]
            recommendation = [item for item in recommendation if item not in bags[i]][:self.n]
            recommendations.append(recommendation)
        return recommendations