from copy import deepcopy
from random import shuffle
from turtle import width
import torch
import torch.nn as nn
import torch.optim as optim
from utils import MyDataset
from src.model import MyNetwork
from tqdm import tqdm
from torch.utils.data import DataLoader
from prepare_data import MyDataset
# import matplotlib.pyplot as plt


def train(dataloader, model, loss_fn, optimizer, device):
    # size = len(dataloader.dataset)
    model.train()
    for batch, (x,y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # log_file.write log
        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(x)
            
            # log_file.write(f"batch {batch}, loss {loss:>7f}  [{current} / {size} ]")
    
def valid(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    valid_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            valid_loss += loss_fn(pred, y).item()
            correct += (abs(pred - y) < 1e-1).type(torch.int).sum().item()
        valid_loss /= num_batches
        correct /= size
        return valid_loss, correct
        

if __name__ == "__main__":
    
    # Get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    datapath = 'data/my_data_4pi'
    
    # Set parameters
    batch_size=512
    width_list=[8,16,32]
    deepth_list=[1, 2]
    lr_list = [0.8]
    epochs = 100
    activate_fn='tanh'

    # Load dataset 
    dataset = torch.load(datapath)
    training_data = dataset['train']
    valid_data = dataset['valid']

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last = True)
    valid_Dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last = True)

    for deepth in deepth_list:
        for width in width_list:
            for lr in lr_list:
                
                log_file = open(f'log/train_log_wd{width}_dp{deepth}_lr{lr}_{activate_fn}','w+')
                print(f'train_log_wd{width}_dp{deepth}_lr{lr}_{activate_fn}')

                for x, y in valid_Dataloader:
                    log_file.write(f"Shape of x: {x.shape}")
                    log_file.write(f"Shape of y: {y.shape}")
                    break

                # Initial model
                model = MyNetwork(batch_size=batch_size,width=width,deepth=deepth)
                loss_fn = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(),lr=lr)
                # loss_set = []
                # idx = [i for i in range(epochs)]
                
                for id in tqdm(range(epochs)):
                    log_file.write(f'Epoch {id+1} \n-------------------------------\n')
                    train(train_dataloader, model, loss_fn, optimizer, device)
                    valid_loss, correct = valid(train_dataloader, model, loss_fn, device)
                    # loss_set.append(valid_loss)
                    log_file.write(f"Avg valid loss {valid_loss:>7f}, Accuracy: {(100*correct):>0.01f}% \n")
                log_file.write("Training done!")

                # plt.plot(idx,loss_set)
                # plt.savefig('loss.pvg')
                # Save model
                torch.save(model.state_dict(),f"model/model_wd{width}_dp{deepth}_lr{lr}_{activate_fn}")
                log_file.write("Saved PyTorch Model State to model.pth")
                
                log_file.close()
    