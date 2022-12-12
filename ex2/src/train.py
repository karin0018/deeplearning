import torch
import os
import torch.nn as nn
import torch.optim as optim
from model import LSTM
from tqdm import tqdm
from torchtext.legacy import data
from torchtext.vocab import Vectors
import sys

def train(dataloader, model, loss_fn, optimizer, device):
    # size = len(dataloader.dataset)
    model.train()
    for _, batch in enumerate(dataloader):
        x, y = batch.Text.to(device), batch.Label.to(device)-1
        
        # Compute prediction error
        pred = model(x)
        # because vocab label set 0 -> 1, 1 -> 2
        loss = loss_fn(pred, y)
    
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def valid(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset) 
    model.eval()
    valid_loss, correct = 0, 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            x, y = batch.Text.to(device), batch.Label.to(device)
            y=y-1
            pred = model(x)
            valid_loss += loss_fn(pred, y).data
            pred = torch.argmax(pred.data,dim=1)
            correct += pred.eq(y.data).sum()
        valid_loss = valid_loss.type(torch.FloatTensor)
        correct = correct.type(torch.FloatTensor)
        valid_loss /= size
        correct /= size
        return valid_loss, correct
        

if __name__ == "__main__":
    
    # Get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    datapath="../dataset/"
    
    # Set parameters
    batch_size = 1024 # batchsize: 16,64,128,256
    hiddim_list = [128,256] # hidden_dimension of LSTM: 128,256
    deepth_list = [1, 2] # one-layer LSTM or two-layer LSTM: 1, 2
    lr_list = [1e-3,1e-4]
    epochs = 100
    bidirectional = False # unidirectional LSTM or bidirectional LSTM: True, False
    
    # Load dataset 
    TEXT = data.Field(lower=True, batch_first=True, fix_length=100)
    LABEL = data.Field(sequential=False)
    train_set, valid_set, test_set = data.TabularDataset.splits(
                                                    path=datapath, train='train.csv', validation='valid.csv', test='test.csv', 
                                                    format='csv', fields=[('Text',TEXT),('Label',LABEL)])
    
    # construct and load word-vectors from a self trained file
    vectors = Vectors(name="../model/word2vec.txt")
    
    TEXT.build_vocab(train_set, vectors = vectors, max_size=10000, min_freq=10)
    LABEL.build_vocab(train_set)
    
    print(TEXT.vocab.freqs.most_common(20))
    print(LABEL.vocab.freqs)
    

    # Create data iteration.
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_set, valid_set, test_set), batch_size=batch_size, device=device, shuffle=True, sort=False)
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    train_iter.repeat = False
    test_iter.repeat = False
    
    log_file = open(f'../log/train_log','w+')
                
    
    for num_layers in deepth_list:
        for hidden_size in hiddim_list:
            for lr in lr_list:
                log_file.write("\n------------------------------------------------------\n")
                log_file.write(f'numlayers{num_layers}_hiddensize{hidden_size}_lr{lr}\n')

                # Initial model
                model = LSTM(vocab_size=len(TEXT.vocab.stoi),pad_idx=pad_idx,embed_size=200,
                             hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional,dropout=0,batch_first=True,labels=2)
                model.embedding.weight.data = TEXT.vocab.vectors
                # frozen pretrained embedding weights
                model.embedding.weight.requires_grad = False 
                model = model.to(device)
                
                train_loss_fn = nn.CrossEntropyLoss()
                valid_loss_fn = nn.CrossEntropyLoss(reduction='sum')
                optimizer = optim.Adam(model.parameters(),lr=lr)
                
                for id in tqdm(range(epochs)):
                    train(train_iter, model, train_loss_fn, optimizer, device)
                    
                    valid_loss, correct = valid(valid_iter, model, valid_loss_fn, device)
                    # loss_set.append(valid_loss)
                
                log_file.write(f"Avg valid loss {valid_loss:>7f}, Accuracy: {(100*correct):>0.01f}% \n")
                log_file.write("Training done!\n")
                
                test_loss, test_correct = valid(test_iter, model, valid_loss_fn, device)
                log_file.write(f"Avg test loss {test_loss:>7f}, Accuracy: {(100*test_correct):>0.01f}% \n")
                

                torch.save(model.state_dict(),f"../model/model_numlayers{num_layers}_hiddensize{hidden_size}_lr{lr}")
                log_file.write("Saved PyTorch Model State to model.pth")
                
    log_file.close()
    