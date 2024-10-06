from sklearn.metrics import recall_score, make_scorer, accuracy_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_classification_scoring(testing:bool=False) -> dict:
    """Return scoring functions for classification task

    Args:
        testing (bool, optional): in test mode. Defaults to False.

    Returns:
        dict: a list of pairs of metric:function
    """
    if not testing:
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'sensitivity': make_scorer(recall_score),
            'specificity': make_scorer(recall_score,pos_label=0),
            'f1': make_scorer(f1_score),
            'auc': make_scorer(roc_auc_score),
            'ppv': make_scorer(precision_score, pos_label=1),
            'npv': make_scorer(precision_score, pos_label=0),
        }
    else:
        scoring = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'sensitivity': recall_score,
            'specificity': recall_score,
            'f1': f1_score,
            'auc': roc_auc_score,
            'ppv': precision_score,
            'npv': precision_score
        }
    return scoring

def get_regression_scoring(testing:bool=False) -> dict:
    """Return scoring functions for regression task

    Args:
        testing (bool, optional): in test mode. Defaults to False.

    Returns:
        dict: a list of pairs of metric:function
    """
    if not testing:
        scoring = {
            'r2': make_scorer(r2_score),
            'rmse': make_scorer(mean_squared_error, squared=False, greater_is_better=False),
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        }
    else:
        scoring = {
            'r2': r2_score,
            'rmse': mean_squared_error,
            'mae': mean_absolute_error,
        }
    return scoring
