
from collections import Counter
from itertools import combinations, product
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_frequency_of_words(lists):
    counter = Counter()
    for elem in tqdm(lists):
        counter.update(elem)
    return counter

def plot_counter(counter, name):
    labels, values = zip(*counter.most_common(50))
    indexes = np.arange(len(labels))
    width = 1
    plt.figure(figsize=(15,15))
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.savefig(name)


def from_counter_occurences_to_counter_frequencies(counter):
    counter_occurences = Counter(list(counter.values()))
    number_of_words = sum(list(counter_occurences.values()))
    counter_frequencies = Counter()
    for key in counter.keys():
        counter_frequencies[key] = counter[key]/number_of_words
    return counter_frequencies
    

def create_dataset(param, lists):
    counter = get_frequency_of_words(lists)
    words_kept = counter.most_common(param.vocabulary_size + param.remove_top_n_words)[param.remove_top_n_words:]
    
    dictionnary_word_to_id = {}
    for i, elem in enumerate(words_kept):
        dictionnary_word_to_id[elem[0]] = i

    #New lists with remaining words
    new_lists = list()
    for elem in tqdm(lists):
        new_sub_list = [word for word in elem if word in dictionnary_word_to_id]
        new_lists.append(new_sub_list)
    lists = new_lists
    
    triplets = list()
    for k, elem in enumerate(lists):
        for i in range(len(elem)-(1+param.window_size)):
            for elem1, elem2 in combinations(elem[i:i+(1 + param.window_size)], 2):
                triplets.append([dictionnary_word_to_id[elem1], k, dictionnary_word_to_id[elem2]])
    
    dictionnary_id_to_word = {v: k for k, v in dictionnary_word_to_id.items()}

    return lists, dictionnary_word_to_id, dictionnary_id_to_word, np.array(triplets)


def generate_batch_data(param):
    if (param.index + param.batch_size < param.number_of_training_pairs):
        batch = param.triplets[param.index : param.index + param.batch_size] 
        param.index += param.batch_size
        return batch
    else:
        batch = param.triplets[param.index:]
        param.index = 0
        return batch

#counter_occurences = Counter(list(counter.values()))
#counter_frequencies = from_counter_occurences_to_counter_frequencies(counter_occurences)
#plot_counter(counter_frequencies, "body_frequencies")




