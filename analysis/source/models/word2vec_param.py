import numpy as np
import random
from read_in_and_cleaning import read_in_and_clean, retrieve_word_seq_text

class Param(object):

    def __init__(self, text_var, fake_subset=None):

        self.seed = 123
        random.seed(self.seed)

        self.text_var = text_var
        self.fake_subset = fake_subset
        if fake_subset==None:
            self.file_name = "word2vec_{}".format(self.text_var)
        if fake_subset==0:
            self.file_name = "word2vec_{}_true".format(self.text_var)
        if fake_subset==1:
            self.file_name = "word2vec_{}_fake".format(self.text_var)

        self.data = read_in_and_clean(fake_subset=self.fake_subset)
        self.text = retrieve_word_seq_text(self.data, self.text_var, lemmatize=True)

        self.max_vocabulary_size = 20000 # Total number of different words in the vocabulary
        self.remove_top_n_words = 0
        self.window_size = 3
        self.word_embedding_size = 200

        self.training_steps = 50000
        self.index = 0
        self.batch_size = 128
        self.num_sampled = int(50)
        self.learning_rate = 2.5e-3

        self.print_loss_every = 1000
        self.print_valid_every = 5000
        self.save_embeddings_every = 5000
        self.print_most_common = 25 # Print most common words in the data
        self.nb_eval_words = 25
