import pandas as pd
import numpy as np
import json
from models.visualization_utils import loss_comparison, get_reduced, word_comparison
import random
random.seed(123)

# Read in data
df_loss_true = pd.read_csv('data/word2vec_body_true_loss_values.csv')
df_loss_fake = pd.read_csv('data/word2vec_body_fake_loss_values.csv')
dict_true = json.load(open('data/word2vec_body_true_dict.json'))
dict_fake = json.load(open('data/word2vec_body_fake_dict.json'))
df_true = pd.DataFrame(np.load('data/word2vec_body_true.npy'))
df_fake = pd.DataFrame(np.load('data/word2vec_body_fake.npy'))

# Plot loss
loss_comparison(df_loss_true, df_loss_fake)

# Generate Reduced Form
reduced_true = get_reduced(df_true, dict_true, method='pca')
reduced_true.to_csv('data/word2vec_body_true_pca.csv', index=False)
reduced_fake = get_reduced(df_fake, dict_fake, method='pca')
reduced_fake.to_csv('data/word2vec_body_fake_pca.csv', index=False)

# Plot Word Comparisons
word_comparison(reduced_true, reduced_fake, words=["i", "you", "we"], file_name='pronouns')
word_comparison(reduced_true, reduced_fake, words=["cnn", "msnbc", "fox"], file_name='news_outlets')
word_comparison(reduced_true, reduced_fake, words=["russian", "french", "german", "chinese", "british"], file_name='nationalities')
word_comparison(reduced_true, reduced_fake, words=["massachusetts", "texas", "arizona", "dakota", "oregon"], file_name='states')
word_comparison(reduced_true, reduced_fake, words=["monday", "tuesday", "wednesday", "thursday", "friday"], file_name='week_days')
word_comparison(reduced_true, reduced_fake, words=["hillary", "crooked"], file_name='crooked_hillary')
word_comparison(reduced_true, reduced_fake, words=["president", "leader", "puppet"], file_name='president_role')
word_comparison(reduced_true, reduced_fake, words=["obama", "trump", "clinton", "cruz", "sander"], file_name='politician_names')
word_comparison(reduced_true, reduced_fake, words=["isi", "muslim", "terrorist"], file_name='muslims')
word_comparison(reduced_true, reduced_fake, words=["tax", "return", "foundation", "fraud"], file_name='taxes')
word_comparison(reduced_true, reduced_fake, words=["democracy", "liberalism", "capitalism", "socialism", "liberal"], file_name='liberalism')

# word_comparison(reduced_true, reduced_fake, words=["birth", "certificate", "fake"], file_name='birther')
# word_comparison(reduced_true, reduced_fake, words=["climate", "science", "water", "atmosphere"], file_name='climate')
# word_comparison(reduced_true, reduced_fake, words=["jew", "jewish", "millionaire", "cabal", "billionaire", "million", "billion"], file_name='jewish')
# word_comparison(reduced_true, reduced_fake, words=["robby", "mook", "manager", "strategist", "campaign"], file_name='mook')
# word_comparison(reduced_true, reduced_fake, words=["democratic", "socialist"], file_name='dem_socialist')
# word_comparison(reduced_true, reduced_fake, words=["socialism", "insanity", "infamy"], file_name='socialism')
# word_comparison(reduced_true, reduced_fake, words=["ohio", "pennsylvania", "michigan", "wisconsin", "indiana"], file_name='midwest_states')
# word_comparison(reduced_true, reduced_fake, words=["fake", "news"], file_name='fake_news')
# word_comparison(reduced_true, reduced_fake, words=["science", "medicine", "magic"], file_name='science')
# word_comparison(reduced_true, reduced_fake, words=["health", "medicine", "remedy", "unbelievable"], file_name='medical')
# word_comparison(reduced_true, reduced_fake, words=["climate", "hoax", "farming", "fossil"], file_name='climate_2')
