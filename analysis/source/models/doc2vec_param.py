
import numpy as np
from read_in_and_cleaning import load_data_and_clean, get_all_labels, get_all_headlines, get_all_bodies, get_dico_by_id
from models.doc2vec_utils import create_dataset

class Param(object):

    def __init__(self):

        self.data = load_data_and_clean()
        self.data_by_id = get_dico_by_id(self.data, 10)
        self.bodies = get_all_bodies(self.data_by_id)
        self.labels = get_all_labels(self.data_by_id)
        np.save("labels.npy", self.labels)
        print("Labels saved")
        
        self.vocabulary_size = 3500
        self.user_size = len(self.bodies)
        self.remove_top_n_words = 100
        self.window_size = 1
        print("Parameters")
        print("vocab_size", self.vocabulary_size, "user_size", self.user_size)

        self.lists, self.dictionnary_word_to_id, self.dictionnary_id_to_word, self.triplets = create_dataset(self, self.bodies)
        np.random.shuffle(self.triplets)
        self.number_of_training_pairs = len(self.triplets)

        self.valid_ids = np.array([10, 50, 250, 500, 1000])
        

        self.training_steps = 20000
        self.index = 0
        self.batch_size = 128
        self.num_sampled = int(50)
        self.learning_rate = 2.5e-3
        self.print_loss_every = 200
        self.print_valid_every = 2500
        self.save_embeddings_every = 2500
        self.word_embedding_size = 150
        self.doc_embedding_size = 150


        
