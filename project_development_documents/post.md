# Text Analysis and Fake News: An Exploratory Look

## Comparing TF–IDF and Doc2Vec for Basic Fake News Detection

The flagging of fake news has become a major issue for social networks and internet platforms. Though human moderators and crowdsourcing techniques are mostly employed to combat the spread of misinformation, the hope for some is that Natural Language Processing and AI will soon be capable of identifying fake news immediately, <a href="https://www.washingtonpost.com/news/the-switch/wp/2018/04/11/ai-will-solve-facebooks-most-vexing-problems-mark-zuckerberg-says-just-dont-ask-when-or-how/" target="_blank">as Mark Zucherberg told the US Congress last year</a>.

In this project, we take an existing dataset of real and fake news entries, and look at whether they present different text structures. The data we use was compiled by George McIntire in 2017, and used by the author to build a Baysian classification model of fake news (the results for published on <a href="https://opendatascience.com/how-to-build-a-fake-news-classification-model/" target="_blank">OpenDataScience.com</a>). The data and model used to be available on <a href="https://github.com/GeorgeMcIntire/fake_real_news_dataset" target="_blank">this GitHub repository</a>, but it has appartently been pulled down recently. You can download the data as we used it <a href="https://www.dropbox.com/s/tonh5b1rn9iz77s/fake_or_real_news.csv?dl=0" target="_blank">here</a>.

<!-- While George McIntire uses -->

We use the **`word2vec`** sequence model to assign two high-dimensionality vectors to every word: one in the True News corpus and one among the Fake News entries. We then plot the words according to the two first principal components of the data, and observe the relative positions of the same words in both corpi. Our main findings are that, even though fake news does have some of the same structural logic as real news, there are some important differences, especially on political or polarizing terms. This contributes to the important body of research on trying to

### Data



### Modeling

The modeling method we use is **`word2vec`**, a common Natural Language Processing model that assigns a high-dimensionality vector to every word depending on sequential similarities with other words. `word2vec` considers every word within a sliding window.

**Ugo, can you add some piching here, please?**

### Results

- Magic-science
- Soros-Funded
- States clustered together
- Hillary-?
- “Pope Francis shocks world, endorses Donald Trump for president”
- “Donald Trump sent his own plane to transport 200 stranded marines”
- #Pizzagate: underground human trafficking/child sex abuse ring
- “WikiLeaks confirms Hillary sold weapons to ISIS … Then drops another bombshell”
- “FBI agent suspected in Hillary email leaks found dead in apartment murder-suicide”
- “FBI director received millions from Clinton Foundation, his brother’s law firm does Clinton’s taxes”
- “ISIS leader calls for American Muslim voters to support Hillary Clinton”
- “Hillary Clinton in 2013: ‘I would like to see people like Donald Trump run for office; they’re honest and can’t be bought’”


<!-- In this project, I take an existing dataset of real and fake news entries, and test algorithmic classifications based on two different features and machine learning models: word-frequency **TF--IDF** features and **`doc2vec`** features based on word and document embeddings. The results demonstrate that both methods perform similarly on the dataset, with the more elementary TF--IDF method actually outperforming the `doc2vec` in most cases.

The data we use is George McIntyre

In particular, I create two text sets of features for every corpus entry:
- **TF--IDF**: I create trivial "Term Frequency--Inverse Document Frequency" (TF--IDF) features by calculating the number of times each word appears in an article, and normalizing by the number of times of word appears across all articles. In the words of Wikipedia: "TF--IDF [...] is intended to reflect how important a word is to a document in a collection or corpus". In this case, the document's text is treated as a "bag of words", with no consideration of word ordering. A article's vector is defined as its TF--IDF score for every word of the corpus of documents.
- **`doc2vec`**: I also run a sequence model that assigns a high-dimensional vector to every article in the corpus. This method is an extension of **`word2vec`**, which assigns vectors to words based on the context they appear in. `doc2vec` creates similar word vectors, but with an additional document vector as well. For a more detailed description of the technique, check out <a href="https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e" target="_blank">this blog posts</a>, for example. -->
