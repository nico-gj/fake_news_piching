
import numpy as np
from read_in_and_cleaning import load_data_and_clean, get_all_labels, get_all_headlines, get_all_bodies
from models.doc2vec_utils import create_dataset

class Param(object):

    def __init__(self):

        self.data = load_data_and_clean()
        self.headlines = get_all_headlines(self.data)
        self.labels = get_all_labels(self.data)
        self.bodies = get_all_bodies(self.data)
        self.vocabulary_size = 4000
        self.user_size = len(self.bodies)
        self.remove_top_n_words = 75
        self.window_size = 1
        self.lists, self.dictionnary_word_to_id, self.dictionnary_id_to_word, self.triplets = create_dataset(self, self.bodies)
        self.number_of_training_pairs = len(self.triplets)

        self.valid_ids = np.array([10, 50, 250, 500, 1000])
        

        self.training_steps = 10000
        self.index = 0
        self.batch_size = 128
        self.num_sampled = int(50)
        self.learning_rate = 1e-3
        self.print_loss_every = 200
        self.print_valid_every = 1000
        self.save_embeddings_every = 1000
        self.word_embedding_size = 150
        self.doc_embedding_size = 150


        
