import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import time
import math
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from torch.multiprocessing import Pool, Process, set_start_method
import matplotlib.pyplot as plt

def data_load(PATH):
    train =np.loadtxt(f'{PATH}/vina_train.csv', delimiter=",", dtype=np.float32)
    x_train = torch.tensor(train[:, 1:])
    y_train = torch.tensor(train[:, :1])
    length = x_train.size(1)
    print(length)
    df = TensorDataset(x_train,y_train)
    dataloader = DataLoader(df, batch_size=2048*4, shuffle=True)
    print("data loading")
    test = np.loadtxt(f'{PATH}/vina_test.csv', delimiter=",", dtype=np.float32)
    x_test = torch.tensor(test[:, 1:])
    y_test = torch.tensor(test[:, :1])
    df_t = TensorDataset(x_test,y_test)
    dataloader_t = DataLoader(df_t, batch_size=2048*4, shuffle=True)
    return dataloader, dataloader_t,length


class Net(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.3):
        super(Net, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Linear1 = nn.Linear(self.input_size, 1024)
        self.Linear2 = nn.Linear(1024, 528)
        self.Linear3 = nn.Linear(528, self.output_size)
        self.Batch1 = nn.BatchNorm1d(self.input_size)
        self.Drop = nn.Dropout(p=dropout)

    def forward(self, input_tensor):
        x = self.Batch1(input_tensor)
        x = self.Drop(x)
        x = F.elu(self.Linear1(x))
        x = F.elu(self.Linear2(x))
        x = self.Linear3(x)
        return x


def train(train_data, test_data, model, device,epochs=200):
    learning_rate = 0.001
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses=[]
    test_losses=[]
    for epoch in range(epochs):
        running_loss = 0.0
        running_loss_ = 0.0
        epoch_step = 0
        model.train()
        for i, (xx,yy) in enumerate(train_data):
            #GPU 전송
            xx = xx.to(device=device)
            yy = yy.to(device=device)
            #train start
            train_output = model(xx)
            loss = criterion(train_output, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_step += 1
        train_losses.append(running_loss)
        if epoch % 10 == 0:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(f'Train loss at {epoch} is {running_loss/epoch_step}')
            epoch_step = 0
            model.eval()
            for i, (xx,yy) in enumerate(test_data):
                with torch.no_grad():
                    x_test = xx.to(device=device)
                    y_test = yy.to(device=device)
                    #신경망 평가모드로 설정 & test data 손실함수 계산
                    test_output = model(x_test)
                    test_loss = criterion(test_output, y_test)
                    running_loss_ += test_loss.item()
                    epoch_step += 1
            test_losses.append(test_loss)
            print(f'test loss at {epoch} is {running_loss_/epoch_step}')
            #correlation coeff
            #new_shape = (len(y_test), 1)
            #y_test = y_test.view(new_shape)
            vx = y_test - torch.mean(y_test) #true - true mean
            vy = test_output - torch.mean(test_output) #pred - pred mean
            corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))  # use Pearson correlation
            print(f'correlation coefficient at {epoch} is {corr}')
            running_loss = 0.0

def test(test_data, model):
    running_loss = 0.0
    criterion = nn.MSELoss()
    epoch_step = 0
    true = []
    pred = []
    test_loss=[]
    #model.eval()
    for i, (xx,yy) in enumerate(test_data):
        with torch.no_grad():
            xx = xx.to(device=device)
            yy = yy.to(device=device)
            test_output = model(xx)
            loss = criterion(test_output, yy)
            running_loss += loss.item()
            epoch_step += 1
            for i,x in enumerate(test_output):
                x = x.item()
                pred.append(x)
            for a,b in enumerate(yy):
                b = b.item()
                true.append(b)
            
    print('last loss at {}'.format(running_loss/epoch_step))
if __name__ == '__main__':
    try:
         set_start_method('spawn')
    except RuntimeError:
        pass
    device = torch.device("cuda")
    dict_PATH = "/home/choi/docking/pytorch/surechem_vina"
    num_process = 10
    pool = Pool(num_process)
    print("data loading start")
    train_data, test_data, length = data_load(dict_PATH)
    print(f"length = {length}") #input size
    model = Net(length, 1, dropout=0.3).to(device=device)
    print(model)

    #model.load_state_dict(torch.load("/home/choi/docking/pytorch/surechem_vina_gpu/l3_pytorch_model_multi.pt",map_location=device))
    processes = []
    for rank in range(num_process):
        p = Process(target=train, args=(train_data,test_data,model,device,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    test(test_data, model)
    torch.save(model.state_dict(), '/home/choi/docking/pytorch/surechem_vina_gpu/l3_pytorch_model_multi_520.pt')
