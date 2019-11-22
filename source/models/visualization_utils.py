import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from adjustText import adjust_text
import random
random.seed(123)

double_fig_size=(10, 5)

# Plot loss
def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(df, max_val=8, n=50):
    losses = moving_average(np.array(df['loss_value']), n)
    losses = losses[0::n]
    iterations = np.array(df['iteration'])[0::n]
    plt.scatter(iterations, losses, s=5)
    plt.xlabel('iteration')
    plt.ylabel('word2vec loss value')
    plt.title('{} News'.format(type))
    plt.ylim(0, max_val)

def loss_comparison(df_loss_true, df_loss_fake):

    fig = plt.figure(figsize=(10,3))

    fig.add_subplot(1,2,1)
    plot_loss(df_loss_true)
    plt.title('True News')

    fig.add_subplot(1,2,2)
    plot_loss(df_loss_fake)
    plt.title('Fake News')

    plt.tight_layout()
    plt.savefig('output/loss_functions.png', bbox_inches='tight')

def get_reduced(df, dict, method):
    if method=='pca':
        pca = PCA(n_components=2)
        pca.fit(df)
        reduced = pd.DataFrame(pca.transform(df))
    elif method=='tsne':
        tsne = TSNE(n_components=2)
        reduced = pd.DataFrame(tsne.fit_transform(df))
    reduced.rename(columns={0:'axis_1', 1:'axis_2'}, inplace=True)
    reduced['word']= reduced.index.astype(str).map(dict)
    return reduced


def plot_word_scatter(df, words):
    df = df[(df['axis_1']>-5)&(df['axis_1']<5)&(df['axis_2']>-5)&(df['axis_2']<5)]
    plt.scatter(df['axis_1'], df['axis_2'], s=5, c='grey')
    subset = df[df['word'].isin(words)].reset_index(drop=True)
    plt.scatter(subset['axis_1'], subset['axis_2'], s=50, c='red')
    plt.xlabel('PCA Axis 1')
    plt.ylabel('PCA Axis 2')

    texts = []
    for x, y, s in zip(subset['axis_1'], subset['axis_2'], subset['word']):
        texts.append(plt.text(x, y, s, size='medium', fontweight='bold'))
    adjust_text(texts)

def word_comparison(reduced_true, reduced_fake, words, file_name):
    fig = plt.figure(figsize=double_fig_size)

    for word in words:
        if word not in list(reduced_true['word']):
            print('{} not in True data.'.format(word))
        if word not in list(reduced_fake['word']):
            print('{} not in Fake data.'.format(word))

    fig.add_subplot(1,2,1)
    plot_word_scatter(reduced_true, words)
    plt.title('True News Articles')

    fig.add_subplot(1,2,2)
    plot_word_scatter(reduced_fake, words)
    plt.title('Fake News Articles')

    plt.tight_layout()
    plt.savefig('output/word_scatters/{}_scatter.png'.format(file_name), bbox_inches='tight')
    return
