import random 
import torch
import math 


class MyDataset(torch.utils.data.Dataset):
    """自定义数据集"""
    def __init__(self, x,y):
        self.x, self.y = x,y
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)
    

def sinx_ex(start, end, num_samples):
    """生成 y = sinx + e^(-x) 的数据集"""
    # x = torch.empty(num_samples,dtype=torch.float32).uniform_(start,end)
    x = torch.linspace(start,end,num_samples) # torch.Size([10000])
    # # print(x.size())
    x = torch.unsqueeze(x,dim=1) # torch.Size([10000, 1])
    # # print(x.size())
    y = torch.sin(x) + torch.exp(-x)
    my_data = MyDataset(x,y)
    
    train_size = int(num_samples*0.6)
    valid_size = int(num_samples*0.2)
    test_size = valid_size
    train_data, valid_data, test_data = torch.utils.data.random_split(
                                                                        dataset = my_data,
                                                                        lengths = [train_size,valid_size,test_size], 
                                                                        generator=torch.Generator().manual_seed(0)
                                                                    )
    data_set = {'train':train_data,'valid':valid_data,'test':test_data}


    torch.save(data_set, 'data/my_data_4pi')
    print('data size = ', num_samples, 'train valid test ',len(train_data),len(valid_data),len(test_data),'saved in ','my_data')


sinx_ex(0.0, 4*torch.pi, 10000)

