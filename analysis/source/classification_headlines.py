
from collections import Counter
from itertools import combinations, product
import tensorflow as tf
from read_in_and_cleaning import load_data_and_clean, get_all_labels, get_all_headlines, get_all_bodies
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

data = load_data_and_clean()
headlines = get_all_headlines(data)
labels = get_all_labels(data)
bodies = get_all_bodies(data)
vocab_size = 5000
remove_top_n_words = 75
window_size = 1

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
    number_of_words = sum(list(counter_occurences.values()))
    counter_frequencies = Counter()
    for key in counter.keys():
        counter_frequencies[key] = counter[key]/number_of_words
    return counter_frequencies
    

#counter = get_frequency_of_words(headlines)
#counter_occurences = Counter(list(counter.values()))
#counter_frequencies = from_counter_occurences_to_counter_frequencies(counter_occurences)
#plot_counter(counter_frequencies, "headline_frequencies")

def create_dataset(lists):
    counter = get_frequency_of_words(lists)
    words_kept = counter.most_common(vocab_size+remove_top_n_words)[remove_top_n_words:]
    
    dictionnary_word_to_id = {}
    for i, elem in enumerate(words_kept):
        dictionnary_word_to_id[elem[0]] = i
    
    pairs = list()
    for elem in tqdm(lists):
        for i in range(len(elem)-(1+window_size)):
            for elem1, elem2 in combinations(elem[i:i+(1+window_size)], 2):
                pairs.append([dictionnary_word_to_id[elem1], dictionnary_word_to_id[elem2]])
    print(len(pairs))

create_dataset(bodies)

#counter_occurences = Counter(list(counter.values()))
#counter_frequencies = from_counter_occurences_to_counter_frequencies(counter_occurences)
#plot_counter(counter_frequencies, "body_frequencies")
