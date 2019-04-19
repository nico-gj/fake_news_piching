##################################################
## Python Setup
##################################################
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

seed = 123

from read_in_and_cleaning import read_in_and_clean

##################################################
# Def Unsupervised Functions
##################################################

def pca_and_plot(df_dict, text, features, hue_vars):

    path = "data/{}_{}.npy".format(features, text)
    df = pd.DataFrame(np.load(path))

    pca = PCA(n_components=2)
    pca.fit(df)

    pca = pd.DataFrame(pca.transform(df))
    pca.rename(columns={0:'pca_axis_1', 1:'pca_axis_2'}, inplace=True)
    pca['id'] = df_dict['id'].values()
    for hue_var in hue_vars:
        pca[hue_var] = df_dict[hue_var].values()

    ### Plot:
    if features == 'doc2vec':
        title_desc = "Embedding Vector Coordinates"
    elif features == 'tfidf':
        title_desc = "TF-IDF on N-Grams"
    else:
        title_desc = ""

    for hue_var in hue_vars:
        sns.set(font_scale=1.3)
        pca.sort_values(hue_var, inplace=True)
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.scatterplot(ax=ax, x=pca['pca_axis_1'], y=pca['pca_axis_2'], hue=pca[hue_var], s=25)
        ax = plt.gca()
        ax.set_title("Principal Component Analysis of {}".format(title_desc),  {'fontsize': 25})
        plt.savefig('output/pca_{}_{}_{}.png'.format(features, text, hue_var), bbox_inches='tight')

    return pca

def lda(text, n_topics, n_top_words, seed=seed):

    path = "data/tfidf_{}.npy".format(text)
    tfidf = pd.DataFrame(np.load(path))

    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=seed)
    doctopic = lda.fit_transform(tfidf)

    ls_keywords = []
    for topic in enumerate(lda.components_):
        word_idx = np.argsort(topic)[::-1][:n_top_words]
        keywords = 'Topic {}: '.format(i)
        keywords += ', '.join(features[j] for j in word_idx)
        ls_keywords.append(keywords)

    return ls_keywords, doctopic

##################################################
# Load in Data:
##################################################
df_dict = read_in_and_clean()
print("{} observations for analysis".format(len(df_dict[list(df_dict.keys())[0]])))

##################################################
# PCA:
##################################################

pca_body_tfidf = pca_and_plot(df_dict=df_dict, text='body', features='tfidf', hue_vars=['label'])
pca_headline_tfidf = pca_and_plot(df_dict=df_dict, text='headline', features='tfidf', hue_vars=['label'])
pca_body_doc2vec = pca_and_plot(df_dict=df_dict, text='body', features='doc2vec', hue_vars=['label'])
pca_headline_doc2vec = pca_and_plot(df_dict=df_dict, text='headline', features='doc2vec', hue_vars=['label'])

##################################################
# K-Means on PCA
##################################################



##################################################
# LDA
##################################################

# lda(text='body', n_topics=5, n_top_words=10)
