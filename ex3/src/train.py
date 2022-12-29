import torch
import torch.nn as nn
import torch.optim as optim
from prepare_data import ImdbDataset
from model import Bert
from tqdm import tqdm
import logging
import copy
from torch.utils.data import DataLoader


def train(dataloader, model, loss_fn, optimizer, device):
    # size = len(dataloader.dataset)
    model.train()
    for _, (x, y) in enumerate(dataloader):
        x, y = x, y.to(device)
        # Compute prediction error
        pred = model(x)
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
        for _, (x,y) in enumerate(dataloader):
            x, y = x, y.to(device)
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

    datapath="/data/lvrui/deeplearning/ex2/dataset/"
    
    # Set parameters
    batch_size = 64 # batchsize: 16,64,128,256
    length_list = [128,256] # sequence length
    lr_list = [1e-5,1e-6]
    epochs = 15

    # Load dataset 
    trainData = ImdbDataset(mode='train', path=datapath)
    validData = ImdbDataset(mode='valid',path=datapath)
    testData = ImdbDataset(mode='test',path=datapath)
    
    # creat data iteration
    train_Dataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
    valid_Dataloader = DataLoader(validData, batch_size=batch_size, shuffle=True)
    test_Dataloader = DataLoader(testData, batch_size=batch_size, shuffle=True)

    
    logging.basicConfig(
        filename='../log/train_test.txt',
        format= '%(message)s',
        level=logging.INFO,
        filemode='a'
    )
        
    
    for max_length in length_list:
        for lr in lr_list:
            logging.info("------------------------------------------------------")
            logging.info(f'length_list{max_length}_lr{lr}')
            
            # Initial model
            model = Bert(max_length=max_length, device=device)
            model = model.to(device)
                
            train_loss_fn = nn.CrossEntropyLoss()
            valid_loss_fn = nn.CrossEntropyLoss(reduction='sum')
            optimizer = optim.AdamW(model.parameters(),lr=lr)
            
            best_valid_loss = 0
            best_acc = 0
            best_model = None
            for id in tqdm(range(epochs)):
                # logging.info(f"------ epoch {id} ------")
                train(train_Dataloader, model, train_loss_fn, optimizer, device)
                    
                valid_loss, correct = valid(valid_Dataloader, model, valid_loss_fn, device)
                
                if correct > best_acc:
                    best_acc = correct
                    best_valid_loss = valid
                    best_model = copy.deepcopy(model)
                
            logging.info(f"Best avg valid loss {best_valid_loss:>7f}, Accuracy: {(100*best_acc):>0.01f}% ")
            logging.info("Training done!")  
            test_loss, test_correct = valid(test_Dataloader, best_model, valid_loss_fn, device)
            logging.info(f"Avg test loss {test_loss:>7f}, Accuracy: {(100*test_correct):>0.01f}% ")

            torch.save(best_model.state_dict(),f"../model/length_list{max_length}_lr{lr}_acc{test_correct}")
            logging.info(f"Saved PyTorch Model State to /model/length_list{max_length}_lr{lr}_acc{test_correct}")
                

    