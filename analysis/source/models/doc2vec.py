
from models.doc2vec_utils import generate_batch_data
import tensorflow as tf 
import numpy as np
import os


def create_and_train_doc2vec_model(param):
    # Start a graph session
    sess = tf.Session()
    print('Creating Model')
    
    # Define Embeddings:
    word_embeddings = tf.Variable(tf.random_uniform([param.vocabulary_size, param.word_embedding_size], -1.0, 1.0))
    doc_embeddings = tf.Variable(tf.random_uniform([param.user_size, param.doc_embedding_size], -1.0, 1.0))
    
    # NCE loss parameters
    nce_weights = tf.Variable(tf.truncated_normal([param.vocabulary_size, param.word_embedding_size + param.doc_embedding_size], stddev=1.0 / np.sqrt(param.word_embedding_size + param.doc_embedding_size)))
    nce_biases = tf.Variable(tf.zeros([param.vocabulary_size]))

    # Create data/target placeholders
    word_inputs = tf.placeholder(tf.int32, shape=[None])
    doc_inputs = tf.placeholder(tf.int32, shape=[None])
    word_targets = tf.placeholder(tf.int32, shape=[None,1])
    valid_dataset = tf.constant(param.valid_ids, dtype=tf.int32)

    # Lookup the word embedding
    # Add together element embeddings in window:
    embed_words = tf.nn.embedding_lookup(word_embeddings, word_inputs)
    embed_docs = tf.nn.embedding_lookup(doc_embeddings, doc_inputs)
    final_embed = tf.concat([embed_words, embed_docs], axis=-1)
    
    # Get loss from prediction
    #loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, word_targets, final_embed, param.num_sampled, param.vocabulary_size))
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(nce_weights, nce_biases, word_targets, final_embed, param.num_sampled, param.vocabulary_size))

    # Create optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=param.learning_rate)
    train_step = optimizer.minimize(loss)

    # Cosine similarity between words
    norm = tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keepdims=True))
    normalized_embeddings = word_embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Create model saving operation
    saver = tf.train.Saver({"word_embeddings": word_embeddings, "doc_embeddings": doc_embeddings})

    # Add variable initializer.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Run the skip gram model.
    print('Starting Training')
    loss_vec = []
    for i in range(param.training_steps):
        batch = generate_batch_data(param)
        word_i, doc_i, word_t = batch[:,0], batch[:,1], np.reshape(batch[:,2], (-1,1))
        feed_dict = {word_inputs: word_i, doc_inputs : doc_i, word_targets: word_t}

        # Run the train step
        _, loss_val = sess.run([train_step, loss], feed_dict=feed_dict)
        loss_vec.append(loss_val)

        # Return the loss
        if (i + 1) % param.print_loss_every == 0:
            loss_val = sess.run(loss, feed_dict=feed_dict)
            print('50 last losses at step {} : {}'.format(i + 1, sum(loss_vec[-50:])/50))

        # Validation: Print some random words and top 5 related words
        if (i + 1) % param.print_valid_every == 0:
            sim = sess.run(similarity, feed_dict=feed_dict)
            for j in range(len(param.valid_ids)):
                valid_word = param.dictionnary_id_to_word[param.valid_ids[j]]
                top_k = 5  # number of nearest neighbors
                nearest = (-sim[j, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to {}:".format(valid_word)
                for k in range(top_k):
                    close_word = param.dictionnary_id_to_word[nearest[k]]
                    log_str = '{} {},'.format(log_str, close_word)
                print(log_str)

        # Save dictionary + embeddings
        if (i + 1) % param.save_embeddings_every == 0:
            np.save("doc_embeddings.npy", sess.run(doc_embeddings))