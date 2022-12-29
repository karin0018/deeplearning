import os
import re
import pandas as pd
from torch.utils.data import Dataset


class ImdbDataset(Dataset):

    def __init__(self, mode, path="../../dataset"):
        """Dataset initial

        Args:
            mode (string): train, valid or test

        Returns:
            None
        """
        self.data = pd.read_csv(path + '/' + mode + '.csv', header=None, names=['review','label'])
        
    def clear(self, text):
        # replace the unuse tokens to " "
        filters = ['!','"','#','$','%','&','\(','\)','\*','\+',',','-','\.','/',':',';','<','=','>','\?','@'
            ,'\[','\\','\]','^','_','`','\{','\|','\}','~','\t','\n','\x97','\x96','”','“',]
        text = re.sub("<.*?>"," ",text,flags=re.S)	
        text = re.sub("|".join(filters)," ",text,flags=re.S)
        
        return text
    
    def __getitem__(self, index):
        
        sentence, label = self.data['review'][index], self.data['label'][index]
        
        sentence = self.clear(sentence)
    
        return sentence, label

    def __len__(self):
        return len(self.data)
    
# reference: https://blog.csdn.net/raelum/article/details/127568409
def read_imdb(path='../dataset/aclImdb', is_train=True):
    print("read imdb dataset ...")
    reviews, labels = [], []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(path, 'train' if is_train else 'test',
                                   label)
        for filename in os.listdir(folder_name):
            with open(os.path.join(folder_name, filename),
                      mode='r',
                      encoding='utf-8') as f:
                reviews.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    print("read done!")
    return reviews, labels

# reference: https://www.cnblogs.com/jiangxinyang/p/10207273.html

def load_imdb(path="../dataset/aclImdb", data_path="../dataset"):
    print('load imdb dataset ...')
    reviews, labels = read_imdb(path, True)

    train_all_set = pd.DataFrame({'review': reviews, 'label': labels})
    # random seed is 0, train/valid = 7/3
    train_set = train_all_set.sample(frac=0.7, random_state=0, axis=0)
    valid_set = train_all_set[~train_all_set.index.isin(train_set.index)]
    train_set.to_csv(data_path + "/train.csv", index=False, header=False)
    valid_set.to_csv(data_path + "/valid.csv", index=False, header=False)

    reviews, labels = read_imdb(path, False)
    test_set = pd.DataFrame({'review': reviews, 'label': labels})

    test_set.to_csv(data_path + "/test.csv", index=False, header=False)

    print('all dataset have been prepared in ' + data_path +
          '/train,valid,test')

if __name__ == "__main__":
    load_imdb()

