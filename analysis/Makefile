##################################
## Create Features
##################################

data/tfidf_body.csv data/tfidf_body.npy data/tfidf_headline.csv data/tfidf_headline.npy: source/main_tfidf.py source/models/tfidf.py source/read_in_and_cleaning.py data/raw/*
	python source/main_tfidf.py

data/doc2vec_body.csv data/doc2vec_body.npy data/doc2vec_headline.csv data/doc2vec_headline.npy: source/main_doc2vec.py source/models/doc2vec.py source/models/doc2vec_utils.py source/models/doc2vec_param.py source/read_in_and_cleaning.py data/raw/*
	python source/main_doc2vec.py

##################################
## Exploratory LogReg and Tree
##################################

output/exploratory_*: source/exploratory_regressions.py data/tfidf_*.csv
	python source/exploratory_regressions.py

##################################
## Unsupervised Classification
##################################
output/unsupervised_*: source/unsupervised_classification.py data/tfidf_*.npy data/doc2vec_*.npy
	python source/unsupervised_classification.py

##################################
## Supervised Classification
##################################
output/model_*: source/supervised_classification.py data/tfidf_*.npy data/doc2vec_*.npy
	python source/supervised_classification.py

##################################
## Supervised Classification
##################################
all: output/exploratory_* output/unsupervised_* output/model_* 
