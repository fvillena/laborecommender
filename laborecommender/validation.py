import statistics
import sklearn.metrics
import sklearn.model_selection
from . import data
import numpy as np

def confusion_matrix(true: list,predicted: list) -> tuple:
    """
    Constructs a recommender system pseudo confusion matrix from true and predicted lists.

    Parameters
    ----------
    true : list of str
        List of true elements.
    predicted : list of str
        List of predicted elements.
    
    Returns
    -------
    n_items_we_recommend : int
        Amount of items the recommender system recommended.
    n_of_our_recommendations_that_are_relevant : int
        Amount of recommended items that are the true list of elements.
    n_of_all_the_possible_relevant_items : int
        Amount of elements in the true list of elements
    
    Examples
    --------
    >>> import laborecommender.validation
    >>> true = ["a","b","c","d"]
    >>> predicted = ["b","c","d","e"]
    >>> laborecommender.validation.confusion_matrix(true,predicted)
    (4, 3, 4)

    """
    recommendations_that_are_relevant = [test for test in predicted if test in true]
    n_items_we_recommend = len(predicted)
    n_of_our_recommendations_that_are_relevant = len(recommendations_that_are_relevant)
    n_of_all_the_possible_relevant_items = len(true)
    return (n_items_we_recommend,n_of_our_recommendations_that_are_relevant,n_of_all_the_possible_relevant_items)

def precision(true: list,predicted: list) -> float:
    """
    Computes the precision metric of a given true and predicted list.

    Parameters
    ----------
    true : list of str
        List of true elements.
    predicted : list of str
        List of predicted elements.

    Returns
    -------
    float
        Precision metric.

    """
    n_items_we_recommend,n_of_our_recommendations_that_are_relevant,_ = confusion_matrix(true,predicted)
    p = n_of_our_recommendations_that_are_relevant / n_items_we_recommend
    return p
def recall(true: list,predicted: list) -> float:
    """
    Computes the recall metric of a given true and predicted list.

    Parameters
    ----------
    true : list of str
        List of true elements.
    predicted : list of str
        List of predicted elements.

    Returns
    -------
    float
        Recall metric.

    """
    _,n_of_our_recommendations_that_are_relevant,n_of_all_the_possible_relevant_items = confusion_matrix(true,predicted)
    r = n_of_our_recommendations_that_are_relevant / n_of_all_the_possible_relevant_items
    return r

def average_metric(true: list,predicted: list,metric) -> float:
    """
    Computes an average metric from a list of true and predicted values.

    This function iterates from 1 to `len(predicted)` where in each iteration
    computes the metric over the true and `predicted[:current_index]`, at the end
    an average metric is calculated.

    Parameters
    ----------
    true : list of str
        List of true elements.
    predicted : list of str
        List of predicted elements.
    metric : callable
        A metric function.
    
    Returns
    -------
    float
        Average metric.

    """
    metrics = []
    for k in range(1,len(predicted)+1):
        predicted_k = predicted[:k]
        p = metric(true,predicted_k)      
        metrics.append(p)
    try:
        return statistics.mean(metrics)
    except statistics.StatisticsError:
        return 0.0

def mean_average_metric(y_true: list,y_pred: list,metric) -> float:
    """
    Computes the mean average metric of a list of lists of true and predicted values.

    This function iterates over each true and predicted bag of laboratory tests and computes
    the average metric of the true and predicted pair. At the and an average of the metrics
    is calculated.

    Parameters
    ----------
    true : list of list of str
        List of true elements.
    predicted : list of list of str
        List of predicted elements.
    metric : callable
        A metric function.
    
    Returns
    -------
    float
        Mean verage metric.

    """
    average_metrics = []
    for t,p in zip(y_true,y_pred):
        average_metrics.append(average_metric(t,p,metric))
    return statistics.mean(average_metrics)

def mean_average_precision(true: list,predicted: list) -> float:
    """
    Computes mean average precision metric from a list of lists of true and predicted values.

    This is a function that wraps `laborecommender.validation.mean_average_metric` with
    `laborecommender.validation.precision` metric.

    Parameters
    ----------
    true : list of list of str
        List of true elements.
    predicted : list of str
        List of predicted elements.
    
    Returns
    -------
    float
        Mean average precision.
    """
    def average_precision(true,predicted):
        return average_metric(true,predicted,precision)
    return mean_average_metric(true,predicted,average_precision)

def mean_average_recall(true,predicted):
    """
    Computes mean average recall metric from a list of lists of true and predicted values.

    This is a function that wraps `laborecommender.validation.mean_average_metric` with
    `laborecommender.validation.recall` metric.

    Parameters
    ----------
    true : list of list of str
        List of true elements.
    predicted : list of str
        List of predicted elements.
    
    Returns
    -------
    float
        Mean average recall.
    """
    def average_recall(true,predicted):
        return average_metric(true,predicted,recall)
    return mean_average_metric(true,predicted,average_recall)

def mean_average_f_beta(true,predicted,beta):
    """
    Computes mean average f beta metric from a list of lists of true and predicted values.

    Parameters
    ----------
    true : list of str
        List of true elements.
    predicted : list of str
        List of predicted elements.
    beta : int
        Beta parameter of the f beta score.

    Returns
    -------
    float
        F beta metric.

    """
    p = mean_average_precision(true,predicted)
    r = mean_average_recall(true,predicted)
    b2 = beta ** 2
    return ( 1 + b2 ) * ( ( p * r ) / ( ( b2 * p ) + r ) )

def mean_average_f_1(true,predicted):
    """
    Computes mean average F1 metric from a list of lists of true and predicted values.

    This is a function that wraps `laborecommender.validation.mean_average_metric` with
    `laborecommender.validation.f_beta` with beta equal to 1 metric.

    Parameters
    ----------
    true : list of list of str
        List of true elements.
    predicted : list of str
        List of predicted elements.
    
    Returns
    -------
    float
        Mean average F1.
    """
    return mean_average_f_beta(true,predicted,1)

def cross_val_score(estimator, X, scoring, cv=5, n=5):
    """
    Computes scores across cross validated subsets.

    Parameters
    ----------
    estimator : instance of `laborecommender.model.LaboRecommender`
        LaboRecommender instance to cross validate.
    X : list of list of str
        Dataset to cross validate with.
    scoring : callable
        Scoring function.
    cv : int, default=5
        Number of k-folds.
    n : int, default=5
        Number of tests to recommend.
    
    Returns
    -------
    list
        List of scores for each fold.

    """
    kf = sklearn.model_selection.KFold(n_splits=cv)
    scores = []
    for train_i, test_i in kf.split(X):
        train = [X[i] for i in train_i]
        test = [X[i] for i in test_i]
        test_x = []
        test_y = []
        for bag in test:
            x,y = data.cut_bag(bag)
            test_x.extend(x)
            test_y.extend(y)
        test_x = test_x[:len(train)]
        test_y = test_y[:len(train)]
        estimator.fit(train)
        predicted = estimator.predict(test_x,n=n)
        scores.append(scoring(test_y,predicted))
    return scores

def grid_search(param_grid: dict,estimator,X,scorer,n=5,cv=5):
    """
    Performs a grid search over a parameter grid.

    Parameters
    ----------
    param_grid : dict
        Param grid
    estimator : instance of `laborecommender.model.LaboRecommender`
        LaboRecommender instance to cross validate.
    X : list of list of str
        Dataset to cross validate with.
    scorer : callable
        Scoring function.
    n : int, default=5
        Number of tests to recommend.
    cv : int, default=5
        Number of k-folds.
    
    Returns
    -------
    dict
        Dictionary of grid search results.
    
    Examples
    --------
    >>> import sklearn.model_selection
    >>> import laborecommender.data
    >>> import laborecommender.validation
    >>> bags = laborecommender.data.get_bags_from_mimic()
    >>> train_bags, test_bags = sklearn.model_selection.train_test_split(bags,random_state=11)
    >>> test_bags_x, test_bags_y = laborecommender.data.make_supervised_dataset(test_bags)
    >>> param_grid = {
            "metric":["jaccard","minkowski"],
            "k":[15,20,40,50,80,100]
        }
    >>> grid_search_results = laborecommender.validation.grid_search(
            param_grid,
            laborecommender.model.LaboRecommender(),
            train_bags,
            laborecommender.validation.mean_average_f_1
        )
    >>> grid_search_results["best_mean_result"]
    0.3667637180287459

    """
    results = {
        "params":[],
        "raw_results":[],
        "mean_results":[]
    }
    for params in sklearn.model_selection.ParameterGrid(param_grid):
        current_estimator = estimator.set_params(**params)
        results["params"].append(params)
        current_results = cross_val_score(current_estimator,X,scorer,n=n,cv=cv)
        results["raw_results"].append(current_results)
        results["mean_results"].append(statistics.mean(current_results))
        best_i = np.argmax(results["mean_results"])
        results["best_params"] = results["params"][best_i]
        results["best_raw_result"] = results["raw_results"][best_i]
        results["best_mean_result"] = results["mean_results"][best_i]
    return results