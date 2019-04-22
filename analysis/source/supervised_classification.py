import matplotlib.pyplot as plt
import seaborn as sns
from models.prediction_models import logistic_regression, decision_tree, boosted_decision_tree, random_forest
from models.prediction_models_param import Param

def plot_roc_curve(roc_df, param, roc_curve_title):
    roc_c = plt.figure()
    plt.plot(roc_df['fpr'], roc_df['tpr'])
    plt.plot([0, 1], [0, 1], linestyle = '--')
    plt.xlabel('False Positive Rate', fontsize = 18)
    plt.ylabel('True Positive Rate', fontsize = 18)
    plt.title(roc_curve_title, fontsize = 18)
    sns.despine()
    plt.savefig('output/model_{}_roc_{}_{}.png'.format(results['model_name'], param.features, param.text_var), bbox_inches='tight')

def export_results(results, param, roc_curve_title=""):
    if 'coef_table' in results.keys():
        coef_table = results['coef_table']
        coef_table.to_csv('output/model_{}_coefs_{}_{}.csv'.format(results['model_name'], param.features, param.text_var), index=False)
    if 'tree' in results.keys():
        tree = results['tree']
        tree = '\n'.join(tree)
        file = open('output/model_{}_tree_{}_{}.txt'.format(results['model_name'], param.features, param.text_var), 'w+')
        file.write(tree)
        file.close()
    if 'roc_df' in results.keys():
        roc_df = results['roc_df']
        plot_roc_curve(roc_df, param, roc_curve_title)

##########################
# Runs on Total Data
##########################

param = Param(features='tfidf', text_var='body')
results = logistic_regression(param)
export_results(results, param, roc_curve_title="Logistic Regression on Body TFIDF ROC Curve")
results = decision_tree(param)
export_results(results, param, roc_curve_title="Decision Tree on Body TFIDF ROC Curve")
results = boosted_decision_tree(param)
export_results(results, param, roc_curve_title="Boosted Decision Tree on Body TFIDF ROC Curve")
results = random_forest(param)
export_results(results, param, roc_curve_title="Random Forest on Body TFIDF ROC Curve")

param = Param(features='tfidf', text_var='headline')
results = logistic_regression(param)
export_results(results, param, roc_curve_title="Logistic Regression on Headline TFIDF ROC Curve")
results = decision_tree(param)
export_results(results, param, roc_curve_title="Decision Tree on Headline TFIDF ROC Curve")
results = boosted_decision_tree(param)
export_results(results, param, roc_curve_title="Boosted Decision Tree on Headline TFIDF ROC Curve")
results = random_forest(param)
export_results(results, param, roc_curve_title="Random Forest on Headline TFIDF ROC Curve")

param = Param(features='doc2vec', text_var='body')
results = logistic_regression(param)
export_results(results, param, roc_curve_title="Logistic Regression on Body Doc2Vec ROC Curve")
results = decision_tree(param)
export_results(results, param, roc_curve_title="Decision Tree on Body Doc2Vec ROC Curve")
results = boosted_decision_tree(param)
export_results(results, param, roc_curve_title="Boosted Decision Tree on Body Doc2Vec ROC Curve")
results = random_forest(param)
export_results(results, param, roc_curve_title="Random Forest on Body Doc2Vec ROC Curve")

param = Param(features='doc2vec', text_var='headline')
results = logistic_regression(param)
export_results(results, param, roc_curve_title="Logistic Regression on Headline Doc2Vec ROC Curve")
results = decision_tree(param)
export_results(results, param, roc_curve_title="Decision Tree on Headline Doc2Vec ROC Curve")
results = boosted_decision_tree(param)
export_results(results, param, roc_curve_title="Boosted Decision Tree on Headline Doc2Vec ROC Curve")
results = random_forest(param)
export_results(results, param, roc_curve_title="Random Forest on Headline Doc2Vec ROC Curve")

##########################
# Runs on First Axes of PCA
##########################

# TO DO!


#############
# Sandbox
#############

# df_dict = read_in_and_clean()

# # There are n columns in the feature matrix
# # after One Hot Encoding.
# data = np.load("data/tfidf_body.npy")
# print("Number features", data.shape[1])
#
# raw_data = read_in_and_clean()
# # raw_data_by_id = get_dico_by_id(raw_data)
# # sources = np.array(get_all_source(raw_data_by_id))
# # keep_data_points = []
# # for i, elem in enumerate(sources):
# #     if (elem == 'CHICAGO TRIBUNE') or (elem == 'NY TIMES'):
# #         keep_data_points.append(i)
#
# data = data[np.array(keep_data_points)]
# sources = sources[np.array(keep_data_points)]
# labels = np.array([(1,0) if elem=="CHICAGO TRIBUNE" else (0,1) for elem in sources])
#
# def train_classif(data, labels):
#
#     length_data, features_data = data.shape
#     training_ratio = 0.70
#     shuffle = np.random.permutation(length_data)
#     data, labels = data[shuffle], labels[shuffle]
#     training_data, test_data = data[:int(training_ratio*length_data)], data[int(training_ratio*length_data):]
#     training_labels, test_labels = labels[:int(training_ratio*length_data)], labels[int(training_ratio*length_data):]
#     length_training_data = len(training_data)
#
#     learning_rate = 2.5e-3
#     epochs = 401
#     batch_size = 128
#     index = 0
#
#     def generate_batch_data(index):
#         if (index + batch_size < length_training_data):
#             x = training_data[index : index + batch_size]
#             y = training_labels[index : index + batch_size]
#             index += batch_size
#             return x, y, index
#         else:
#             x = training_data[index:]
#             y = training_labels[index:]
#             index = 0
#             return x, y, index
#
#     X = tf.placeholder(tf.float32, [None, features_data])
#     Y = tf.placeholder(tf.float32, [None, 2])
#     W = tf.Variable(tf.zeros([features_data, 2]))
#     b = tf.Variable(tf.zeros([2]))
#
#     Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))
#     loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_hat, labels = Y))
#     optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(loss)
#     correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
#         cost_history, accuracy_history = [], []
#         for epoch in range(epochs):
#             x, y, index = generate_batch_data(index)
#             loss_val, _ = sess.run([loss, optimizer], feed_dict = {X : x, Y : y})
#             #if epoch % 100 == 0:
#             #    training_accuracy = accuracy.eval({X : training_data[:1000], Y : training_labels[:1000]}) * 100
#             #    test_accuracy = accuracy.eval({X : test_data, Y : test_labels}) * 100
#             #    print("Epoch", epoch, "Training Acc", training_accuracy, "Test Acc", test_accuracy)
#         return accuracy.eval({X : test_data, Y : test_labels}) * 100
#
#
# def run_logreg_classif(df_dict, features, text):
#
#     data = np.load("data/{}_{}.npy".format(features, text))
#     labels = np.array(list(df_dict['label'].values()))
#
#     print("Number of features:", data.shape[1])
#
#     accuracies = list()
#     for i in range(25):
#         accuracies.append(train_classif(data, labels))
#     accuracies = np.array(accuracies)
#     print("Conf int", np.mean(accuracies)-0.5*np.std(accuracies), np.mean(accuracies)+0.5*np.std(accuracies))
#
#     return accuracies
