import pandas as pd
import numpy as np
import itertools
import warnings
import matplotlib.pyplot as plt

from six.moves import zip, range
from sklearn.model_selection import train_test_split
import sklearn.feature_extraction.text as f_e
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoLars
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, roc_curve, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import preprocessing
from sklearn import model_selection as sk_ms
import nltk
from nltk import SnowballStemmer

from collections import Counter, OrderedDict

from read_in_and_cleaning import read_in_and_clean

def create_bag_of_words(corpus, ngram_range = (1, 1), stop_words = None, stem = False, min_df = 0.05, max_df = 0.95, use_idf = False):

    #parameters for vectorizer
    ANALYZER = "word" #unit of features are single words rather then phrases of words
    STRIP_ACCENTS = 'unicode'

    if stem:
        stemmer = nltk.SnowballStemmer("english")
        tokenize = lambda x: [stemmer.stem(i) for i in x.split()]
        stop_words = [tokenize(x)[0] for x in stop_words]
    else:
        tokenize = None
    vectorizer = CountVectorizer(analyzer=ANALYZER, tokenizer=tokenize, ngram_range=ngram_range, stop_words = stop_words, strip_accents=STRIP_ACCENTS, min_df = min_df, max_df = max_df)

    bag_of_words = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()

    if use_idf:
        NORM = None #turn on normalization flag
        SMOOTH_IDF = True #prvents division by zero errors
        SUBLINEAR_IDF = True #replace TF with 1 + log(TF)
        transformer = TfidfTransformer(norm = NORM,smooth_idf = SMOOTH_IDF,sublinear_tf = True)
        #get the bag-of-words from the vectorizer and
        #then use TFIDF to limit the tokens found throughout the text
        tfidf = transformer.fit_transform(bag_of_words)

        return tfidf, features
    else:
        return bag_of_words, features


def get_word_counts(bag_of_words, feature_names):

    # convert bag of words to array
    np_bag_of_words = bag_of_words.toarray()

    # calculate word count.
    word_count = np.sum(np_bag_of_words,axis=0)

    # convert to flattened array.
    np_word_count = np.asarray(word_count).ravel()

    # create dict of words mapped to count of occurrences of each word.
    dict_word_counts = dict(zip(feature_names, np_word_count))

    # Create ordered dictionary
    orddict_word_counts = OrderedDict( sorted(dict_word_counts.items(), key=lambda x: x[1], reverse=True), )

    return orddict_word_counts

def tf_idf(text, min_df=0.05, max_df=0.95, stop_words=None, stem=True, ngram_range=(1,2), use_idf=True):

    ## Data Import
    df_dict = read_in_and_clean()
    corpus = np.array(list(df_dict['{}'.format(text)].values()))

    # Create TFIDF Features:
    corpus_tfidf, corpus_features = create_bag_of_words(corpus, stop_words=stop_words, min_df=min_df, max_df=max_df, stem=stem, ngram_range=ngram_range, use_idf=use_idf)

    df = pd.DataFrame(corpus_tfidf.toarray())
    df.columns = corpus_features

    print("{}: {} features".format(text, df.shape[1]))

    with open('data/tfidf_{}.npy'.format(text), 'wb') as np_file:
        np.save(np_file, df)
