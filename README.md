# Overview
This is a framework for ML-based analysis. This kind of analysis aims to focus more on finding the insights, as well as correlations of the data rather than benchmarking the ML models. Therefore, a standard framework to quickly get the insights of the dataset is needed.

 The common steps are as follows:

1. Preprocessing (not yet in this repo). Note that some types of preprocessing must be in the second step, such as mean imputation or standardization.
2. Define the preprocess pipeline (e.g. drop correlated features, over/undersample, reduce dimension) and param grid for the preprocessing methods in the config file.
3. List models we are going to try and their corresponding hyperparam grids in the config file.
4. Run the analysis.

# Analysis Framework
The analysis steps are:

1. <u>Execute nested cross validation</u>: This is not to choose a set of optimal hyperparams, but to evaluate the whole pipeline. For example, it is useful for checking the stability/variation of the predictions and optimized hyper-params.
2. <u>Cross validate</u>: Choose the optimal hyper-params, retrain with train set and report results on test set.
3. <u>Feature importance</u>: The last step is to determine features that are highly related and have impacts on the performance of the models. Now only SHAP method is available.