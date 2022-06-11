from more_itertools import last
from Linear.Linear_GPU import Linear
from utils.utils import FoodFeeling, lr_schedular
import torch
from torch.utils.data import DataLoader
from os.path import exists
import math

def get_args(n_features):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    training_params ={
        'learning_rate': 0.01,
        'batch_size': 50,
        'epochs':1,
        'device': device,
        'n_features': n_features 
    }
    return training_params
    data = FoodFeeling(has_bias=True)
    data_loader = DataLoader(data, batch_size=batch_size)
    input_dim = data.n_features 

def load_model(training_params):
    input_dim = training_params['n_features']
    learning_rate = training_params['learning_rate']
    batch_size = training_params['batch_size']
    device = training_params['device']
    MAX_LOSS = 1e5

    model = Linear(input_dim=input_dim, lr=learning_rate, batch_size=batch_size, device=device).to(device)
    best_params = torch.ones_like(model.parameters())
    best_params.copy_(model.parameters())
    best_loss = MAX_LOSS

    if exists("checkpoint/linear.pth"):
        last_model = torch.load('checkpoint/linear.pth')
        # print(last_model)
        model.load_weight(last_model['params'])
        best_loss = last_model['loss']
        best_params = last_model['params']
        # print('Loading E:',model.parameters())
        # print('Loading Okay')
        # print(best_loss)
    
    return model, best_params, best_loss

def save_model(best_params, best_loss):
    params = {
        'params': best_params,
        'loss': best_loss
    }
    torch.save(params, "checkpoint/linear.pth")

def train():
    
    data = FoodFeeling(has_bias=True)
    training_params = get_args(data.n_features)
    data_loader = DataLoader(data, batch_size=training_params['batch_size'])
    # input_dim = data.n_features 
    epochs = training_params['epochs'] 
    device = training_params['device']
    model, best_params, best_loss = load_model(training_params=training_params)
    print(model.parameters(), best_params, best_loss)
    for epoch in range(epochs):
        total_loss = 0
        total_samples = 0
        model.learning_rate = lr_schedular(cur_epoch=epoch+1, lr=model.learning_rate, lr_decay=0.0001, epoch_decay=250)
        for input_data, output_data in data_loader:

            model.zero_grad()
            output_data = output_data.view(1, -1).to(device)
            input_data = torch.transpose(input_data, 0, 1).to(device)
            y_hat = model(input_data)
            total_loss += model.MSELoss(y_hat, output_data)

            total_samples += output_data.shape[0]
            model.train(input_data, output_data)
        if math.isnan(total_loss) == False and best_loss > total_loss/total_samples:
            best_loss = total_loss/total_samples
            best_params.copy_(model.parameters())
        
        print(f'{epoch} with loss mean {total_loss/total_samples}')
    save_model(best_params=best_params, best_loss=best_loss)

if __name__ == '__main__':
    train()