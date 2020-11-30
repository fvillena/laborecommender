import sklearn.pipeline
import sklearn.neighbors
import sklearn.base
import numpy as np
from . import features
import collections
import itertools

def list_of_bags_to_set(bags):
    return [item[0] for item in collections.Counter(itertools.chain.from_iterable(bags)).most_common()]
def remove_items_from_bag(bag,banned_items):
    return [item for item in bag if item not in banned_items]

class NearestBags(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
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
        recommended_bags = self.bags_[recommended_bag_ids]
        recommendations = map(list_of_bags_to_set,recommended_bags)
        recommendations = [remove_items_from_bag(recommendation,bag)[:self.n] for recommendation,bag in zip(recommendations,bags)]
        return recommendations