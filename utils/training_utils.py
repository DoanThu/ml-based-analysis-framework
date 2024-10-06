import logging
from sklearn.exceptions import ConvergenceWarning
from typing import List, Callable
from utils.metrics_utils import get_classification_scoring, get_regression_scoring
from tqdm import tqdm
from sklearn.base import is_classifier
import sys
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline

ConvergenceWarning('ignore')
logging.basicConfig(format="{asctime} - {levelname} - {message}",
                    stream=sys.stdout,
                    style="{",
                    datefmt="%Y-%m-%d %H:%M",
                    level=logging.DEBUG)


def cross_validate(X: pd.DataFrame, y: pd.DataFrame,
                   repeated_cv_func: Callable, search_cv_func: Callable,
                   pipeline: Pipeline, param_grid: dict, target_metrics: List[str],
                   n_jobs: int = -1, verbose: bool = True) -> dict:
    """ Execute params search using cross validation based on each metric in the input metrics.
    For every single metric, return the best estimator according to it

    Args:
        X (pd.DataFrame): X train
        y (pd.DataFrame): y train
        repeated_cv_func (Callable): cross validation function used (e.g. KFold)
        search_cv_func (Callable): search function to find best params (e.g. GridSearch, RandomSearch)
        pipeline (Pipeline): preprocess + model pipeline
        param_grid (dict): param grid
        target_metrics (List[str]): List of metrics (e.g. auc, accuracy)
        n_jobs (int, optional): n_jobs is usually a param of search function. Defaults to -1.
        verbose (bool, optional): level of logging. There are only 2 levels now. Defaults to True.

    Returns:
        dict: dictionary {metric1: estimator1 with best params, metric2: ...}
    """

    if is_classifier(pipeline.get_params()['model']):
        scoring_dict = get_classification_scoring()
    else:
        scoring_dict = get_regression_scoring()

    best_estimators_dict_by_metric = {}

    for metric in tqdm(target_metrics):
        scoring = scoring_dict[metric]
        if verbose:
            logging.info(f'Finding the best estimator for metric: {metric}')
        grid_search = search_cv_func(pipeline, param_grid, cv=repeated_cv_func,
                                     scoring=scoring, n_jobs=n_jobs,
                                     refit=True)
        grid_search.fit(X, y)
        if verbose:
            logging.info(f"Best score for {metric}: {grid_search.best_score_}")

        best_estimators_dict_by_metric[metric] = grid_search.best_estimator_

    return best_estimators_dict_by_metric


def nested_cross_validation(X: pd.DataFrame, y: pd.DataFrame,
                            repeated_times: int,
                            inner_cv_func: Callable, outer_cv_func: Callable, search_cv_func: Callable,
                            pipeline: Pipeline, param_grid: dict, target_metrics: List[str]):
    """ Perform nested cross validation and return mean +- std of each metric in input metrics

    Args:
        X (pd.DataFrame): X of the entire dataset
        y (pd.DataFrame): y of the entire dataset
        repeated_times (int): number of times to split train test
        inner_cv_func (Callable): cross validation function for the inner loop (note that this is a partial function without random_state param)
        outer_cv_func (Callable): cross validation function for the outer loop (note that this is a partial function without random_state param)
        search_cv_func (Callable): search function to find best params (used for inner loop) (e.g. GridSearch, RandomSearch)
        pipeline (Pipeline): preprocess + model pipeline
        param_grid (dict): param grid
        target_metrics (List[str]): List of metrics (e.g. auc, accuracy)
    """

    if is_classifier(pipeline.get_params()['model']):
        scoring_dict = get_classification_scoring()
    else:
        scoring_dict = get_regression_scoring()

    nested_scores = np.zeros(repeated_times)

    for metric in target_metrics:
        scoring = scoring_dict[metric]
        for i in range(repeated_times):
            inner_cv = inner_cv_func(random_state=i)
            outer_cv = outer_cv_func(random_state=i)
            clf = search_cv_func(pipeline, param_grid,cv=inner_cv, scoring=scoring)
            nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring=scoring)

            nested_scores[i] = nested_score.mean()

        logging.info(metric.upper() + ':' +
                     f'{nested_scores.mean():.4f} +- {nested_scores.std():.4f}')
