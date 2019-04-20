from models.tfidf import tf_idf

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

tf_idf('body', min_df=0.1, max_df=0.9, stop_words=stop_words, stem=True, ngram_range=(1,2), use_idf=True)
tf_idf('headline', min_df=0.01, max_df=0.95, stop_words=stop_words, stem=True, ngram_range=(1,2), use_idf=True)
