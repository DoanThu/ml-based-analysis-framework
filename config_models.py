from sklearn.svm import SVC
svc_name = 'svc'
svc_model = SVC()
svc_param_grid = {
    'model__C': [0.1, 1.0],
    'model__gamma': [0.1, 1.0],
    'model__degree': [2, 3],
    }

from sklearn.linear_model import LogisticRegression
lr_name = 'lr'
lr_model = LogisticRegression()
lr_param_grid = {
    'model__C': [0.1, 1.0, 5.0, 10.0],  
    'model__penalty': ['l1', 'l2'],  
    'model__solver': ['liblinear']
    }

from sklearn.neighbors import KNeighborsClassifier
knc_name = 'knc'
knc_model = KNeighborsClassifier()
knc_param_grid = {
'model__n_neighbors': [3, 5],
'model__weights': ['uniform', 'distance'],
'model__metric': ['l1', 'l2'],
'model__algorithm': ['brute', 'kd_tree']
}

MODELS = [(svc_name, svc_model, svc_param_grid), 
          (lr_name, lr_model, lr_param_grid),
          (knc_name, knc_model, knc_param_grid),
          ]