from pickletools import optimize
import torch
import numpy as np
from utils.utils import FoodFeeling


class Linear():
    def __init__(self, input_dim, output_dim=1, lr=0.01, num_samples=1):
        self.__params = torch.ones(1, input_dim)
        self.learning_rate = lr
        self.n_samples = num_samples

    def __call__(self, input_data):
        if torch.is_tensor(input_data):
            return torch.mm(self.__params, input_data)
        else:
            input_data = torch.tensor(input_data)
            return torch.mm(self.__params, input_data)

    def parameters(self):
        return self.__params

    def MSELoss(self, predicted, true_value):
        loss_list = (predicted - true_value)
        loss = 0
        for row_loss in loss_list:
            loss += pow(row_loss.item(), 2)
        return 1.0/self.n_samples*loss

    def train(self, input_data, output_data):
        y_hat = self.__call__(input_data)
        loss = y_hat - output_data
        loss = loss.item()
        optimizer = self.learning_rate*loss*input_data
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
        output_data = output_data.view(1, 1)
        input_data = input_data.view(-1, 1)
        y_hat = model(input_data)
        total_loss += model.MSELoss(y_hat, output_data)

        total_samples += output_data.shape[0]
        model.train(input_data, output_data)
        print('Data:',input_data, output_data)
        print('Params:', model.parameters())
        break
    break
    print(f'{epoch} with loss mean {total_loss/total_samples}')
