from utils.metrics_utils import get_classification_scoring, get_regression_scoring
from sklearn.base import is_classifier
from imblearn.pipeline import Pipeline
import pandas as pd


def predict_with_estimator(estimator: Pipeline,
                           X_train: pd.DataFrame, y_train: pd.DataFrame,
                           X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """ Given a pipeline and train set, fit the pipeline with the train set and return results performed on test set

    Args:
        estimator (Pipeline): pipeline
        X_train (pd.DataFrame): X in train set
        y_train (pd.DataFrame): y in train set
        X_test (pd.DataFrame): X in test set
        y_test (pd.DataFrame): y in test set

    Returns:
        dict: Return a dictionary {metric1:value1, metric2:value2,...}
    """
    # Re-train
    trained_pl = estimator.fit(X_train, y_train)
    y_pred = trained_pl.predict(X_test)
    
    # Define the scoring metrics
    if is_classifier(estimator.get_params()['model']):
        scoring = get_classification_scoring(testing=True)
    else:
        scoring = get_regression_scoring(testing=True)
        
    scores = {}
    
    # Calculate the scores for each metric
    for metric, scorer in scoring.items():
        add_kwargs = {}
        if metric in ['specificity', 'npv']:
            add_kwargs = {'pos_label': 0}
        scores[metric] = scorer(y_test, y_pred, **add_kwargs)

    res = {}
    for metric, score in scores.items():
        res[metric] = score
    return res