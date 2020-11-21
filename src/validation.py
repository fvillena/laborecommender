import statistics
import sklearn.metrics
import sklearn.model_selection
from . import data
import numpy as np

def confusion_matrix(true,predicted):
    recommendations_that_are_relevant = [test for test in predicted if test in true]
    n_items_we_recommend = len(predicted)
    n_of_our_recommendations_that_are_relevant = len(recommendations_that_are_relevant)
    n_of_all_the_possible_relevant_items = len(true)
    return (n_items_we_recommend,n_of_our_recommendations_that_are_relevant,n_of_all_the_possible_relevant_items)

def precision(true,predicted):
    n_items_we_recommend,n_of_our_recommendations_that_are_relevant,_ = confusion_matrix(true,predicted)
    p = n_of_our_recommendations_that_are_relevant / n_items_we_recommend
    return p
def recall(true,predicted):
    _,n_of_our_recommendations_that_are_relevant,n_of_all_the_possible_relevant_items = confusion_matrix(true,predicted)
    r = n_of_our_recommendations_that_are_relevant / n_of_all_the_possible_relevant_items
    return r

def average_metric(true,predicted,metric):
    metrics = []
    for k in range(1,len(predicted)+1):
        predicted_k = predicted[:k]
        p = metric(true,predicted_k)      
        metrics.append(p)
    try:
        return statistics.mean(metrics)
    except statistics.StatisticsError:
        return 0.0

def mean_average_metric(y_true,y_pred,metric):
    average_metrics = []
    for t,p in zip(y_true,y_pred):
        average_metrics.append(average_metric(t,p,metric))
    return statistics.mean(average_metrics)

def mean_average_precision(true,predicted):
    def average_precision(true,predicted):
        return average_metric(true,predicted,precision)
    return mean_average_metric(true,predicted,average_precision)

def mean_average_recall(true,predicted):
    def average_recall(true,predicted):
        return average_metric(true,predicted,recall)
    return mean_average_metric(true,predicted,average_recall)

def mean_average_f_beta(true,predicted,beta):
    p = mean_average_precision(true,predicted)
    r = mean_average_recall(true,predicted)
    b2 = beta ** 2
    return ( 1 + b2 ) * ( ( p * r ) / ( ( b2 * p ) + r ) )

def mean_average_f_1(true,predicted):
    return mean_average_f_beta(true,predicted,1)

def cross_val_score(estimator, X, scoring, cv=5, n=5):
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

def grid_search(param_grid,estimator,X,scorer,n=5):
    results = {
        "params":[],
        "raw_results":[],
        "mean_results":[]
    }
    for params in sklearn.model_selection.ParameterGrid(param_grid):
        current_estimator = estimator.set_params(**params)
        results["params"].append(params)
        current_results = cross_val_score(current_estimator,X,scorer,n=n)
        results["raw_results"].append(current_results)
        results["mean_results"].append(statistics.mean(current_results))
        best_i = np.argmax(results["mean_results"])
        results["best_params"] = results["params"][best_i]
        results["best_raw_result"] = results["raw_results"][best_i]
        results["best_mean_result"] = results["mean_results"][best_i]
    return results