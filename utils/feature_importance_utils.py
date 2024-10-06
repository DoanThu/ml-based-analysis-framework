import shap
import pandas as pd
from imblearn.pipeline import Pipeline

import shap
def get_shap_values(estimator: Pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame, 
                    X_test: pd.DataFrame, y_test: pd.DataFrame, sample:int=5) -> tuple:
    """ Compute SHAP values based on estimator (pipeline)

    Args:
        estimator (Pipeline): ML pipeline
        X_train (pd.DataFrame): X from train set
        y_train (pd.DataFrame): y from train set
        X_test (pd.DataFrame): X from test set
        y_test (pd.DataFrame): y from test set
        sample (int, optional): number of times sample in SHAP. Defaults to 5.

    Returns:
        obs: X_train after pipline, the number of columns can change if the pipline contains some dimensional operators
        explainer: explainer object of shap
        shap_values: shap values
    """
    
    clf = estimator.fit(X_train, y_train)['model']
    obs = estimator[:-1].fit(X_train, y_train).transform(X_test)
    obs = shap.sample(obs,sample)
    if obs.shape[1] == X_test.shape[1]:
        obs = pd.DataFrame(data=obs, columns=X_test.columns)
    else:
        obs = pd.DataFrame(data=obs, columns=[f'Feature_{i}' for i in range(obs.shape[1])])
    explainer = shap.KernelExplainer(clf.predict, obs)
    shap_values = explainer(obs)
    return obs, explainer, shap_values