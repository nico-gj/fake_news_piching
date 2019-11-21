from models.word2vec_param import Param
from models.word2vec import create_and_train_word2vec_model

param = Param(text_var='body', fake_subset=0)
create_and_train_word2vec_model(param)

param = Param(text_var='body', fake_subset=1)
create_and_train_word2vec_model(param)
