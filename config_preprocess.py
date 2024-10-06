from imblearn.pipeline import Pipeline
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA


PREPROCESS_STEPS = Pipeline([
                ('drop', DropCorrelatedFeatures(threshold=0.8)),
                ('sampling', 'passthrough'), # populated by the param_grid
                ('reduce_dim', 'passthrough'), # populated by the param_grid
                ('scaler', StandardScaler()), 
                ])
# Since imblearn inherits transform method from sklearn, it is suggested that the last step of the preprocessing (not include the model) of the pipeline should also have transform method
# So we can manually transform our test set if needed

PARAM_GRID = {
        'reduce_dim': [PCA(iterated_power=7)],
        'reduce_dim__n_components': [2, 10],
        'sampling': [RandomOverSampler(sampling_strategy=1, random_state=18),
                     RandomUnderSampler(sampling_strategy=1, random_state=18)],
        }