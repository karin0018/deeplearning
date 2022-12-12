import torch
import torch.nn as nn
from model import MyNetwork
from prepare_data import MyDataset
from torch.utils.data import DataLoader

batch_size, epochs = 512, 100
width, deepth = 8, 1
lr = 0.001
activate_fn='tanh'

data_path = 'data/my_data_4pi'
model_path = f'model/model_wd{width}_dp{deepth}_lr{lr}_{activate_fn}'

model = MyNetwork(batch_size=batch_size,width=width)
model.load_state_dict(torch.load(model_path))

model.eval()
loss_fn = nn.MSELoss()
test_data = torch.load(data_path)['test']

test_Dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last = True)

loss = 0
correct = 0
num_batches = len(test_Dataloader)
size = len(test_Dataloader.dataset)
with torch.no_grad():
    for x, y in test_Dataloader:
        pred = model(x)
        loss += loss_fn(pred, y).item()
        correct += (abs(pred - y) < 1e-1).type(torch.int).sum().item()
    loss /= num_batches
    correct /= size
    print(f"test loss: {loss:>7f}, test accuracy: {(100*correct):>0.01f}% ")