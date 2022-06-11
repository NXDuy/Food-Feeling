import torch
import numpy as np
from utils.utils import FoodFeeling
from torch import nn
from torch.utils.data import DataLoader

class Linear(nn.Module):
    def __init__(self, input_dim, output_dim=1, lr=0.01, batch_size=1, device=torch.device('cpu')):
        super(Linear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.__params = torch.ones(1, input_dim).to(device)
        self.learning_rate = lr
        self.grad = torch.ones(batch_size, input_dim).to(device)
        self.grad_zero = True
        self.n_samples = batch_size

    def __call__(self, input_data):
        self.grad_zero = False
        self.grad = torch.transpose(input_data, 0, 1)

        if torch.is_tensor(input_data):
            return torch.mm(self.__params, input_data)
        else:
            input_data = torch.tensor(input_data)
            return torch.mm(self.__params, input_data)

    def zero_grad(self):
        self.grad_zero = True
        self.grad = torch.ones(self.n_samples, self.input_dim).to(self.device)

    def parameters(self):
        return self.__params

    def MSELoss(self, predicted, true_value):
        self.grad_zero = False
        
        loss_list = (predicted - true_value)
        self.grad = torch.mm(loss_list, self.grad) 

        loss = 0
        for row_loss in loss_list:
            for value in row_loss:
                loss += pow(value.item(),2)

        return 1.0/self.n_samples*loss

    def train(self, input_data, output_data):
        if self.grad_zero == True:
            return
        
        optimizer = self.learning_rate*self.grad
        optimizer = optimizer.view(1, -1)
        self.__params -= optimizer

'''
input_dim = 3
learning_rate = 0.01
batch_size = 50 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
data = FoodFeeling()
model = Linear(input_dim=input_dim, lr=learning_rate, batch_size=batch_size).to(device)
data_loader = DataLoader(data, batch_size=batch_size)

epochs = 500
for epoch in range(epochs):
    total_loss = 0
    total_samples = 0
    for input_data, output_data in data_loader:
        # print(input_data.shape, output_data.shape)
        # break
        model.zero_grad()
        # print(output_data.shape)
        output_data = output_data.view(1, -1).to(device)
        # print(output_data.shape)
        input_data = torch.transpose(input_data, 0, 1).to(device)
        # print(input_data.shape, output_data.shape)
        y_hat = model(input_data)
        # print(y_hat.shape)
        # break
        total_loss += model.MSELoss(y_hat, output_data)

        total_samples += output_data.shape[0]
        model.train(input_data, output_data)
        # break
    print(f'{epoch} with loss mean {total_loss/total_samples}')
'''