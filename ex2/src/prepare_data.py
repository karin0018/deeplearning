import os
import torch
import pandas as pd
import gensim
from bs4 import BeautifulSoup
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
# from torchtext import transforms as T
from torch.utils.data import TensorDataset
from gensim.models import word2vec

# reference: https://blog.csdn.net/raelum/article/details/127568409
def read_imdb(path='../dataset/aclImdb', is_train=True):
    print("read imdb dataset ...")
    reviews, labels = [], []
    for label in ['pos','neg']:
        folder_name = os.path.join(path,'train' if is_train else 'test', label)
        for filename in os.listdir(folder_name):
            with open(os.path.join(folder_name,filename),mode='r',encoding='utf-8') as f:
                reviews.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    print("read done!")
    return reviews, labels

# reference: https://www.cnblogs.com/jiangxinyang/p/10207273.html

def cleanReview(subject):
    """clean data

    turn special token to "" and change all word to lower type
    """
    beau = BeautifulSoup(subject)
    newSubject = beau.get_text()
    newSubject = newSubject.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').replace(',', '').replace('.', '').replace('?', '').replace('(', '').replace(')', '')
    newSubject = newSubject.strip().split(" ")
    newSubject = [word.lower() for word in newSubject]
    newSubject = " ".join(newSubject)
    
    return newSubject


def pretrain_dataset(path="../dataset/aclImdb",train_set_path = "../dataset/train_set_word2vec.txt",w2v_path="../model/word2vec"):
    print('pretrain dataset ... ')
    sentences = word2vec.LineSentence(train_set_path)
    # train vector model, set word vector size=200,  train model is skip-gram, and save to .bin
    model = gensim.models.Word2Vec(sentences, vector_size=200, sg=1)  
    model.wv.save_word2vec_format(w2v_path+".txt", binary=False) 
    print('pretrain done! Word2vec model have saved in '+ w2v_path+ '.txt'+' as type of txt.')

def load_imdb(path="../dataset/aclImdb",data_path="../dataset",train_set_path = "../dataset/train_set_word2vec.txt"):
    print('load imdb dataset ...')
    reviews, labels = read_imdb(path,True)
    
    train_all_set = pd.DataFrame({'review':reviews,'label':labels})
    train_all_set['review'] = train_all_set['review'].apply(cleanReview)
    # random seed is 0, train/valid = 7/3
    train_set = train_all_set.sample(frac=0.7,random_state=0,axis=0)
    train_review = train_set['review']
    valid_set = train_all_set[~train_all_set.index.isin(train_set.index)]
    train_review.to_csv(train_set_path,index=False,header=False)
    train_set.to_csv(data_path+"/train.csv",index=False,header=False)
    valid_set.to_csv(data_path+"/valid.csv",index=False,header=False)
    
    reviews, labels = read_imdb(path,False)
    test_set = pd.DataFrame({'review':reviews,'label':labels})
    test_set['review'] = train_all_set['review'].apply(cleanReview)
    
    test_set.to_csv(data_path+"/test.csv",index=False,header=False)
    
    print('all dataset have been prepared in '+ data_path + '/train,valid,test' )

load_imdb()
pretrain_dataset()
    
    
    
    
    
    
    
    
        