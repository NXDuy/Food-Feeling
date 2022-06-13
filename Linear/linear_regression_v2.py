from pickletools import optimize
import torch
import numpy as np
from utils.utils import FoodFeeling
import copy

class Linear():
    def __init__(self, input_dim, output_dim=1, lr=0.01, batch_size=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.__params = torch.ones(output_dim, input_dim)
        self.learning_rate = lr
        self.grad = torch.ones(output_dim, input_dim)
        self.grad_zero = True
        self.n_samples = batch_size
        # self.grad_delta = 0.00001

    def __call__(self, input_data):
        self.grad_zero = False
        self.grad *= torch.transpose(input_data, 0, 1)

        if torch.is_tensor(input_data):
            return torch.mm(self.__params, input_data)
        else:
            input_data = torch.tensor(input_data)
            return torch.mm(self.__params, input_data)

    def zero_grad(self):
        self.grad_zero = True
        self.grad = torch.ones(self.output_dim, self.input_dim)

    def parameters(self):
        return self.__params

    def MSELoss(self, predicted, true_value):
        self.grad_zero = False
        
        loss_list = (predicted - true_value)
        self.grad *= (1.0/self.n_samples*loss_list)

        loss = 0
        for row_loss in loss_list:
            loss += pow(row_loss.item(),2)
        return 1.0/self.n_samples*loss

    def train(self, input_data, output_data):
        if self.grad_zero == True:
            return
        
        optimizer = self.learning_rate*self.grad
        optimizer = optimizer.view(1, -1)
        self.__params -= optimizer


input_dim = 3
learning_rate = 1
data = FoodFeeling()
model = Linear(input_dim=input_dim, lr=learning_rate)

epochs = 100
for epoch in range(epochs):
    total_loss = 0
    total_samples = 0
    for input_data, output_data in data:
        model.zero_grad()
        output_data = output_data.view(1, 1)
        input_data = input_data.view(-1, 1)

        y_hat = model(input_data)
        # print(y_hat.shape, model.grad.shape)
        total_loss += model.MSELoss(y_hat, output_data)

        total_samples += output_data.shape[0]
        model.train(input_data, output_data)
        # print('Data:',input_data, output_data)
        # print('Params:',model.grad, model.parameters())

        # break
    # break
    print(f'{epoch} with loss mean {total_loss/total_samples}')
