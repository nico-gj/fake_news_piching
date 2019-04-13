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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def split_data(param):

    if param.test_perc > 0:
        training_data, test_data = sk_ms.train_test_split(
            param.data,
            train_size = 1 - param.test_perc,
            test_size = param.test_perc,
            shuffle = True,
            random_state = param.seed
        )
    else:
        training_data = param.data
        test_data = pd.DataFrame

    return training_data, test_data

def create_data_matrices(param):

    training_matrix = {}
    test_matrix = {}
    prediction_matrix = {}

    # Figure out the outcome column and the other columns
    out_col = [param.label_name]
    other_cols = [j for j in list(param.training_data.columns) if j != out_col[0]]

    # Get the training matrices
    training_matrix = {
        'X': param.training_data[other_cols].values,
        'y': param.training_data[out_col].values.transpose()[0],
        'variables': other_cols
    }

    # Get the evaluation matrices
    if param.test_perc > 0:
        test_matrix = {
            'X': param.test_data[other_cols].values,
            'y': param.test_data[out_col].values.transpose()[0],
            'variables': other_cols
        }

    # Get the prediction matrices
    if param.prediction_data is not None:
        prediction_matrix = {'X': param.prediction_data[other_cols].values, 'variables':other_cols}

    return training_matrix, test_matrix, prediction_matrix

def get_tree(model, variables, sep = "-    "):

    '''
    Returns a string representation of the decision tree in "tree".
    variables contains the names of the features, and sep contains
    the indentation character(s)
    '''

    print('\nDecision Tree:')

    # Parse the tree structure
    # ------------------------
    # Start by getting the number of nodes
    n_nodes = model.tree_.node_count

    # Create vectors to store node information
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)             # Depth of the node in the tree
    is_leaf = np.zeros(shape=n_nodes, dtype=bool)                    # Whether the node is a leaf
    node_feature = model.tree_.feature                           # Feature splitting at the node
    node_threshold = model.tree_.threshold                       # Threshold for splitting at the node
    node_prefix = [''] * n_nodes                                     # Text to be printed before the node

    # Get the average outcomes at each of the leaves
    node_average = model.tree_.value
    if len(node_average[0][0]) == 1:
        # We have a regressor
        node_average = [i[0][0] for i in node_average]
    else:
        # We have a classifier; find out how many outcomes
        # are positive
        node_average = [i[0][1]*100/sum(i[0]) for i in node_average]

    if variables == None:
        node_feature = [str(i) for i in node_feature]
    else:
        node_feature = [variables[i] for i in node_feature]

    # Create a stack that will contain all nodes left to traverse.
    # Each tuple in the stack will contain the node ID, and its
    # parent's depth. Seed this stack with the root node
    stack = [(0, -1)]

    # Work through the stack
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()

        node_depth[node_id] = parent_depth + 1

        left_child = model.tree_.children_left[node_id]
        right_child = model.tree_.children_right[node_id]

        if left_child == right_child:
            is_leaf[node_id] = 1
        else:
            stack.append((left_child, node_depth[node_id]))
            stack.append((right_child, node_depth[node_id]))

            node_prefix[left_child] = "<= " + str(round(node_threshold[node_id],2)) + " : "
            node_prefix[right_child] = "> " + str(round(node_threshold[node_id],2)) + " : "

    # Prepare the list to print
    out = [ ]
    for i in range(n_nodes):
        if is_leaf[i]:
            out.append( sep * node_depth[i] + node_prefix[i] + "LEAF NODE" )
        else:
            out.append( sep * node_depth[i] + node_prefix[i] + "Split on " + node_feature[i] )

    # Get impurities and number of points
    impurities = [str(round(i, 3)) for i in model.tree_.impurity]
    n_points = [str(i) for i in model.tree_.n_node_samples]
    node_average = [str(round(i, 3)) for i in node_average]

    # Find the maximum lengths
    max_length_tree = max( [len(i) for i in out] ) + 2
    max_length_imp = max( [len(i) for i in impurities] )
    max_length_n = max( [len(i) for i in n_points] )
    max_length_av = max( [len(i) for i in node_average] )
    pad_string = lambda string, size, sep : string + " " + (sep * (size -len(string))) + " "

    for i in range(n_nodes):
        out[i] = pad_string(out[i], max_length_tree, "-") + \
                    pad_string(n_points[i], max_length_n, " ") + \
                    pad_string(impurities[i], max_length_imp, " ") + \
                    pad_string(node_average[i], max_length_av, " ")

    out = [
        pad_string("", max_length_tree, " ")
        + pad_string("n", max_length_n, " ")
        + pad_string("imp", max_length_imp, " ")
        + pad_string("val", max_length_av, " ")
    ] + out

    for i in range(len(out)):
        print(out[i])

    return out


def get_coef_table(param, model):

    coef_table = None
    p_vals_present = False

    variables = param.training_matrix['variables']
    coefs = model.coef_[0]

    if param.binary_data:
        result = sm.Logit(param.training_matrix['y'], param.training_matrix['X']).fit()
    else:
        result = sm.OLS(param.training_matrix['y'], param.training_matrix['X']).fit()

    def pval_to_star(x):
        if x <= 0.001:
            return '***'
        elif x <= 0.01:
            return '**'
        elif x <= 0.05:
            return '*'
        elif x <= 0.1:
            return '.'
        else:
            return ''
    significance = [pval_to_star(x) for x in result.pvalues]

    coef_table = pd.DataFrame({
        'variable': variables,
        'coefficient': coefs,
        'p_values': result.pvalues,
        'significance': significance
    })

    print(coef_table[coef_table['significance']=="***"])

    return coef_table

def evaluate_model(param, model):

    print("\nEvaluating Model...\n")

    if param.binary_data:
        eval_func = roc_auc_score
        eval_method = "AUC"
    else:
        eval_func = r2_score
        eval_method = "R2"

    # Find the in-sample performance
    if param.binary_data:
        training_scored = [x for x in model.predict_proba(param.training_matrix['X'])[:,1]]
    else:
        training_scored = model.predict(param.training_matrix['X'])

    eval_score_training = eval_func(param.training_matrix['y'], training_scored)
    print("Score Training Set: {}".format(eval_score_training))


    if param.test_perc == 0:
        print("No test data to perform evaluation on.")
        results_dict = {}

    else:
        if param.binary_data:
            test_scored = [x for x in model.predict_proba(param.test_matrix['X'])[:,1]]
        else:
            test_scored = model.predict(param.test_matrix['X'])

        eval_score_test = eval_func(param.test_matrix['y'], test_scored)
        print("Score Test Set: {}".format(eval_score_test))

        print("\nConfusion Matrix:")
        if param.binary_data==False:
            print("Not binary data: Confusion Matrix could not be calculated.")
            test_conf_matrix=None
        else:
            test_expected = param.test_matrix['y']
            test_predicted = [1 if x > param.threshold else 0 for x in test_scored]
            test_conf_matrix = confusion_matrix(test_expected, test_predicted)
            print(test_conf_matrix)
            print("Accuracy: {}".format(accuracy_score(test_expected, test_predicted)))
            print("Precision: {}".format(precision_score(test_expected, test_predicted)))
            print("Recall: {}".format(recall_score(test_expected, test_predicted)))

        print("\nROC Curve:")
        if param.binary_data==False:
            print("Not binary data: ROC Curve could not be calculated.")
            fpr, tpr, thresholds = None, None, None
        if param.binary_data:
            print("ROC Curve data exported.")
            # Prepare an ROC curve
            fpr, tpr, thresholds = roc_curve(test_expected, test_scored)

        evaluation_df = pd.DataFrame({
            'true_outcome': param.test_matrix['y'],
            'predicted_score': test_scored,
            'predicted_outcome': test_predicted
        })

        results_dict = {
            'evaluation_df': evaluation_df,
            'fpr':fpr,
            'tpr':tpr,
            'thresholds':thresholds
        }

    return results_dict

def plot_roc(param, results_dict, model_name, title):
    if param.test_perc == 0:
        print("No test data to plot ROC Curve on.")
    else:
        roc_c = plt.figure()
        plt.plot(results_dict['fpr'], results_dict['tpr'])
        plt.plot([0, 1], [0, 1], linestyle = '--')
        plt.xlabel('False Positive Rate', fontsize = 18)
        plt.ylabel('True Positive Rate', fontsize = 18)
        plt.title(title, fontsize = 18)
        sns.despine()
        plt.savefig('output/roc_{}_{}_{}.png'.format(model_name, param.features, param.text), bbox_inches='tight')
