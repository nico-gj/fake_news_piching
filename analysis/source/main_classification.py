import tensorflow as tf
import numpy as np
from read_in_and_cleaning import load_data_and_clean, get_all_labels, get_all_headlines, get_all_bodies, get_dico_by_id, get_all_source

# There are n columns in the feature matrix 
# after One Hot Encoding. 
data = np.load("data/body_embeddings_30000.npy")
print("Number features", data.shape[1])

raw_data = load_data_and_clean()
raw_data_by_id = get_dico_by_id(raw_data)
sources = np.array(get_all_source(raw_data_by_id))
keep_data_points = []
for i, elem in enumerate(sources):
    if (elem == 'CHICAGO TRIBUNE') or (elem == 'NY TIMES'):
        keep_data_points.append(i)

data = data[np.array(keep_data_points)]
sources = sources[np.array(keep_data_points)]
labels = np.array([(1,0) if elem=="CHICAGO TRIBUNE" else (0,1) for elem in sources])


def train_classif(data, labels):
    length_data, features_data = data.shape
    training_ratio = 0.70
    shuffle = np.random.permutation(length_data)
    data, labels = data[shuffle], labels[shuffle]
    training_data, test_data = data[:int(training_ratio*length_data)], data[int(training_ratio*length_data):]
    training_labels, test_labels = labels[:int(training_ratio*length_data)], labels[int(training_ratio*length_data):]
    length_training_data = len(training_data)

    learning_rate = 2.5e-3
    epochs = 401
    batch_size = 128
    index = 0


    def generate_batch_data(index):
        if (index + batch_size < length_training_data):
            x = training_data[index : index + batch_size]
            y = training_labels[index : index + batch_size]
            index += batch_size
            return x, y, index
        else:
            x = training_data[index:]
            y = training_labels[index:]
            index = 0
            return x, y, index


    X = tf.placeholder(tf.float32, [None, features_data]) 
    Y = tf.placeholder(tf.float32, [None, 2])
    W = tf.Variable(tf.zeros([features_data, 2]))
    b = tf.Variable(tf.zeros([2]))

    Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b)) 
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_hat, labels = Y))
    optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

    init = tf.global_variables_initializer()
    with tf.Session() as sess: 
        sess.run(init) 
        cost_history, accuracy_history = [], []
        for epoch in range(epochs): 
            x, y, index = generate_batch_data(index)
            loss_val, _ = sess.run([loss, optimizer], feed_dict = {X : x, Y : y})
            #if epoch % 100 == 0:
            #    training_accuracy = accuracy.eval({X : training_data[:1000], Y : training_labels[:1000]}) * 100
            #    test_accuracy = accuracy.eval({X : test_data, Y : test_labels}) * 100
            #    print("Epoch", epoch, "Training Acc", training_accuracy, "Test Acc", test_accuracy)
        return accuracy.eval({X : test_data, Y : test_labels}) * 100

accuracies = list()
for i in range(25):
    accuracies.append(train_classif(data, labels))
accuracies = np.array(accuracies)
print("Conf int", np.mean(accuracies)-0.5*np.std(accuracies), np.mean(accuracies)+0.5*np.std(accuracies))

