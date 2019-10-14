import collections
from itertools import combinations, product
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def get_frequency_of_words(lists):
    counter = collections.Counter()
    for elem in tqdm(lists):
        counter.update(elem)
    return counter

def create_dataset(param, vocabulary_size):
    counter = get_frequency_of_words(param.text)
    words_kept = counter.most_common(vocabulary_size + param.remove_top_n_words)[param.remove_top_n_words:]

    dictionnary_word_to_id = {}
    for i, elem in enumerate(words_kept):
        dictionnary_word_to_id[elem[0]] = i

    #Lists with remaining words
    lists = list()
    for elem in tqdm(param.text):
        sub_list = [word for word in elem if word in dictionnary_word_to_id]
        lists.append(sub_list)

    triplets = list()
    for k, elem in enumerate(lists):
        for i in range(len(elem)-(1+param.window_size)):
            for elem1, elem2 in combinations(elem[i:i+(1+param.window_size)], 2):
                triplets.append([dictionnary_word_to_id[elem1], k, dictionnary_word_to_id[elem2]])

    dictionnary_id_to_word = {v: k for k, v in dictionnary_word_to_id.items()}

    return lists, dictionnary_id_to_word, np.array(triplets)


def generate_batch_data(param, triplets):
    number_of_training_pairs = len(triplets)
    if (param.index + param.batch_size <= number_of_training_pairs):
        batch = triplets[param.index : param.index + param.batch_size]
    else:
        batch = np.concatenate((triplets[param.index:], triplets[:param.batch_size-(number_of_training_pairs-param.index)]), axis=0)

    param.index = (param.index+param.batch_size)%number_of_training_pairs
    return batch
