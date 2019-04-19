import pandas as pd
import numpy as np
import itertools
import time
import warnings
import patsy as pt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sklearn.feature_extraction.text as f_e
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoLars
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import model_selection as sk_ms
from sklearn.metrics import r2_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from models.prediction_models_utils import get_tree, evaluate_model, get_coef_table


def logistic_regression(param):

    param.alpha = float(param.alpha)

    print("\n\nLOGISTIC REGRESSION\nCreating Model...")

    if param.binary_data:
        if param.alpha == 0:
            model = LogisticRegression(C=np.inf, penalty = 'l2', solver = 'lbfgs', fit_intercept = False)
        else:
            model = LogisticRegression(C=1.0/param.alpha, penalty = 'l1', solver='liblinear', fit_intercept = False)
    else:
        if param.alpha == 0:
            model = LinearRegression(fit_intercept = False)
        else:
            model = LassoLars(alpha = param.alpha, fit_intercept = False)

    model = model.fit(param.training_matrix['X'], param.training_matrix['y'])

    results_dict = evaluate_model(param=param, model=model)

    # Add Coef Table
    coef_table = get_coef_table(param, model)
    results_dict['coef_table'] = coef_table

    results_dict['model_name'] = "logreg"

    return results_dict


def decision_tree(param):

    param.max_depth = int(param.max_depth)

    print("\n\nDECISION TREE\nCreating Model...")

    if param.binary_data:
        model = DecisionTreeClassifier(max_depth = param.max_depth, random_state = param.seed)
    else:
        model = DecisionTreeRegressor(max_depth = max_depth, random_state = seed)

    model = model.fit(param.training_matrix['X'], param.training_matrix['y'])

    results_dict = evaluate_model(param=param, model=model)

    # Add Decision Tree
    tree = get_tree(model=model, variables=param.training_matrix['variables'])
    results_dict['tree'] = tree

    results_dict['model_name'] = "dectree"

    return results_dict


def boosted_decision_tree(param):

    param.max_depth = int(param.max_depth)
    param.n_estimators = int(param.n_estimators)

    print("\n\nBOOSTED DECISION TREE\nCreating Model...")

    if param.binary_data:
        model = GradientBoostingClassifier(max_depth = param.max_depth, random_state = param.seed, n_estimators = param.n_estimators, learning_rate = param.learning_rate)
    else:
        model = GradientBoostingRegressor(max_depth = max_depth, random_state = seed, n_estimators = n_estimators, learning_rate = learning_rate)

    model = model.fit(param.training_matrix['X'], param.training_matrix['y'])

    results_dict = evaluate_model(param=param, model=model)

    results_dict['model_name'] = "boostedtree"

    return results_dict


def random_forest(param):

    param.max_depth = int(param.max_depth)
    param.n_estimators = int(param.n_estimators)
    param.learning_rate = float(param.learning_rate)

    print("\n\nRANDOM FOREST\nCreating Model...")

    if param.binary_data:
        model = RandomForestClassifier(max_depth = param.max_depth, random_state = param.seed, n_estimators = param.n_estimators)
    else:
        model = RandomForestRegressor(max_depth = param.max_depth, random_state = param.seed, n_estimators = param.n_estimators)

    model = model.fit(param.training_matrix['X'], param.training_matrix['y'])

    results_dict = evaluate_model(param=param, model=model)

    results_dict['model_name'] = "randomforest"

    return results_dict
