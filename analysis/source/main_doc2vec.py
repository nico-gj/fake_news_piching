from models.doc2vec_param import Param
from models.doc2vec import create_and_train_doc2vec_model

param = Param(text_var='body')
create_and_train_doc2vec_model(param)

param = Param(text_var='headline')
create_and_train_doc2vec_model(param)
