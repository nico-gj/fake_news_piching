from models.word2vec_utils import get_frequency_of_words, create_dataset, generate_batch_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import json
import random

def create_and_train_word2vec_model(param):

    counter = get_frequency_of_words(param.text)
    vocabulary_size = min(param.max_vocabulary_size, len(counter)-param.remove_top_n_words)
    lists, dictionnary_id_to_word, triplets = create_dataset(param, vocabulary_size)
    np.random.shuffle(triplets)

    print("\nParameters:")
    print("Number of Texts: ", len(param.text))
    print("Words Count: ", len([item for sublist in param.text for item in sublist]))
    print("Unique Words: ", len(counter))
    print("Vocabulary Size: ", vocabulary_size)
    print("Number of Pairs: ", len(triplets))
    print("Most Common Words: ", counter.most_common()[:param.print_most_common])

    valid_ids = np.array(random.sample(range(0, vocabulary_size), param.nb_eval_words))

    # Start a graph session
    sess = tf.Session()
    print('\nCreating Model')

    # Define Embeddings:
    word_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, param.word_embedding_size], -1.0, 1.0))

    # NCE loss parameters
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, param.word_embedding_size], stddev=1.0/np.sqrt(param.word_embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Create data/target placeholders
    word_inputs = tf.placeholder(tf.int32, shape=[None])
    word_targets = tf.placeholder(tf.int32, shape=[None,1])
    valid_dataset = tf.constant(valid_ids, dtype=tf.int32)

    # Lookup the word embedding
    # Add together element embeddings in window:
    embed_words = tf.nn.embedding_lookup(word_embeddings, word_inputs)

    # Get loss from prediction
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(nce_weights, nce_biases, word_targets, embed_words, param.num_sampled, vocabulary_size))

    # Create optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=param.learning_rate)
    train_step = optimizer.minimize(loss)

    # Cosine similarity between words
    norm = tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keepdims=True))
    normalized_embeddings = word_embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Create model saving operation
    saver = tf.train.Saver({"word_embeddings": word_embeddings})

    # Add variable initializer.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Run the skip gram model.
    print('Starting Training')
    loss_vec = []
    for i in range(param.training_steps):
        batch = generate_batch_data(param, triplets)
        word_i, doc_i, word_t = batch[:,0], batch[:,1], np.reshape(batch[:,2], (-1,1))
        feed_dict = {word_inputs: word_i, word_targets: word_t}

        # Run the train step
        _, loss_val = sess.run([train_step, loss], feed_dict=feed_dict)
        loss_vec.append(loss_val)

        # Return the loss
        if (i + 1) % param.print_loss_every == 0:
            loss_val = sess.run(loss, feed_dict=feed_dict)
            print('50 last losses at step {}: {}'.format(i + 1, sum(loss_vec[-50:])/50))

        # Validation: Print some random words and top 5 related words
        if (i + 1) % param.print_valid_every == 0:
            sim = sess.run(similarity, feed_dict=feed_dict)
            for j in range(len(valid_ids)):
                valid_word = dictionnary_id_to_word[valid_ids[j]]
                top_k = 5  # number of nearest neighbors
                nearest = (-sim[j, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to {}:".format(valid_word)
                for k in range(top_k):
                    close_word = dictionnary_id_to_word[nearest[k]]
                    log_str = '{} {},'.format(log_str, close_word)
                print(log_str)

        # Save dictionary + embeddings
        if ((i + 1) % param.save_embeddings_every == 0) or ((i + 1) % param.training_steps == 0):
            np.save("data/{}.npy".format(param.file_name), sess.run(word_embeddings))
            f = open("data/{}_dict.json".format(param.file_name), "w")
            f.write(json.dumps(dictionnary_id_to_word))
            f.close()
