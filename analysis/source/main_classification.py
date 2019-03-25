import tensorflow as tf
import numpy as np

# There are n columns in the feature matrix 
# after One Hot Encoding. 
data = np.load("data/embeddings_data/doc_embeddings.npy")
labels = np.load("data/embeddings_data/doc_labels.npy")
labels = np.array([(1,0) if elem==1 else (0,1) for elem in labels])

#PARAMS LOG REG !!!
length_data, features_data = data.shape
training_ratio = 0.75
shuffle = np.random.permutation(length_data)
data, labels = data[shuffle], labels[shuffle]

training_data, test_data = data[:int(training_ratio*length_data)], data[int(training_ratio*length_data):]
training_labels, test_labels = labels[:int(training_ratio*length_data)], labels[int(training_ratio*length_data):]
length_training_data = len(training_data)

learning_rate = 2.5e-3
epochs = 1500
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

# Hypothesis 
Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b)) 
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_hat, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(loss)
init = tf.global_variables_initializer() 
 

print("Start training")
with tf.Session() as sess: 
    sess.run(init) 
    cost_history, accuracy_history = [], [] 
      
    # Iterating through all the epochs 
    for epoch in range(epochs): 
        cost_per_epoch = 0
          
        x, y, index = generate_batch_data(index)
        loss_val, _ = sess.run([loss, optimizer], feed_dict = {X : x, Y : y})
        cost_history.append(loss_val)
        
        correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y, 1)) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))         
          
        # Displaying result on current Epoch
        if epoch % 100 == 0:
            print("")
            print("Epoch " + str(epoch) + " Cost: " + str(sum(cost_history[-100:])))
            print("Training accuracy", accuracy.eval({X : training_data[:1000], Y : training_labels[:1000]}) * 100)
            print("Test accuracy", accuracy.eval({X : test_data, Y : test_labels}) * 100)
     
    # Final Accuracy 
    correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("\nAccuracy:", accuracy_history[-1], "%") 