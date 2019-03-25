from models.tfidf_word_freq import tf_idf_word_freq

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

tf_idf_word_freq('body', min_df=0.05, max_df=0.95, stop_words=stop_words, stem=True, ngram_range=(1,2), use_idf=True)
tf_idf_word_freq('headline', min_df=0.01, max_df=0.95, stop_words=stop_words, stem=True, ngram_range=(1,2), use_idf=True)
