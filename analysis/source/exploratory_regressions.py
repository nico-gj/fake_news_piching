from models.prediction_models import logistic_regression, decision_tree, boosted_decision_tree, random_forest
from models.prediction_models_param import Param

def export_exploratory_results(results, param):
    if 'coef_table' in results.keys():
        coef_table = results['coef_table']
        coef_table.to_csv('output/exploratory_logreg_coefs_{}_{}.csv'.format(param.features, param.text_var), index=False)
    if 'tree' in results.keys():
        tree = results['tree']
        tree = '\n'.join(tree)
        file = open('output/exploratory_tree_{}_{}.txt'.format(param.features, param.text_var), 'w+')
        file.write(tree)
        file.close()

#############
# Runs
#############

param = Param(features='tfidf', text_var='body', test_perc=0)
results = logistic_regression(param)
export_exploratory_results(results, param)
results = decision_tree(param)
export_exploratory_results(results, param)

param = Param(features='tfidf', text_var='headline', test_perc=0)
results = logistic_regression(param)
export_exploratory_results(results, param)
restults = decision_tree(param)
export_exploratory_results(results, param)
