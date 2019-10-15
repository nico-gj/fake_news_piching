import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
from adjustText import adjust_text

dict = json.load(open('data/word2vec_body_dict.json'))
dict_true = json.load(open('data/word2vec_body_true_dict.json'))
dict_fake = json.load(open('data/word2vec_body_fake_dict.json'))

df = pd.DataFrame(np.load('data/word2vec_body.npy'))
df_true = pd.DataFrame(np.load('data/word2vec_body_true.npy'))
df_fake = pd.DataFrame(np.load('data/word2vec_body_fake.npy'))

def get_pca(df, dict):
    pca = PCA(n_components=2)
    pca.fit(df)
    pca = pd.DataFrame(pca.transform(df))
    pca.rename(columns={0:'pca_1', 1:'pca_2'}, inplace=True)
    pca['word']= pca.index.astype(str).map(dict)
    return pca

pca = get_pca(df, dict)
pca_true = get_pca(df_true, dict_true)
pca_true.to_csv('data/word2vec_pca_true.csv', index=False)
pca_fake = get_pca(df_fake, dict_fake)
pca_fake.to_csv('data/word2vec_pca_fake.csv', index=False)

def word_comparison(words, file_name):
    fig = plt.figure(figsize=(15, 8))

    for word in words:
        if word not in list(pca_true['word']):
            print('{} not in True data.'.format(word))
        if word not in list(pca_fake['word']):
            print('{} not in Fake data.'.format(word))

    def plot_scatter(df, words=words):
        plt.scatter(df['pca_1'], df['pca_2'], s=5, c='grey')
        subset = df[df['word'].isin(words)].reset_index(drop=True)
        plt.scatter(subset['pca_1'], subset['pca_2'], s=100, c='red')
        plt.xlabel('PCA Axis 1')
        plt.xlabel('PCA Axis 2')

        texts = []
        for x, y, s in zip(subset['pca_1'], subset['pca_2'], subset['word']):
            texts.append(plt.text(x, y, s, size='large', fontweight='bold'))
        adjust_text(texts)

    fig.add_subplot(1,2,1)
    plot_scatter(pca_true)
    plt.title('True News Articles')

    fig.add_subplot(1,2,2)
    plot_scatter(pca_fake)
    plt.title('Fake News Articles')

    plt.tight_layout()
    plt.savefig('output/{}_scatter.png'.format(file_name))
    return

word_comparison(["hillary", "crooked"], file_name='crooked_hillary')
word_comparison(["hillary", "email"], file_name='hillary_email')
word_comparison(["climate", "science", "water", "atmosphere"], file_name='climate')
word_comparison(["comey", "liar", "biased", "fbi"], file_name='comey')
word_comparison(["obama", "president", "leader"], file_name='obama')
word_comparison(["obama", "president", "trump", "clinton", "leader"], file_name='president')
word_comparison(["pizza", "child", "trafficking", "human", "code", "food"], file_name='pizzagate')
word_comparison(["pope", "francis", "trump", "endorse", "endorsement", "endorsing", "support", "hillary"], file_name='pope_francis')
word_comparison(["isi", "muslim", "september", "islam", "terrorism", "mosque"], file_name='muslims')
word_comparison(["tax", "return", "foundation", "fraud"], file_name='taxes')
word_comparison(["scientific", "magical", "amazing", "unbelievable"], file_name='science')
word_comparison(["socialism", "democratic", "democrat", "socialist", "sanders", "bernie"], file_name='bernie')
word_comparison(["pocahonta", "warren", "elizabeth", "indian"], file_name='taxes')
