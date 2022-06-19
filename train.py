# from more_itertools import last
from Linear.Linear_GPU import Linear
from utils.utils import FoodFeeling, lr_schedular
import torch
from torch.utils.data import DataLoader, random_split
from os.path import exists
import math
import matplotlib.pyplot as plt

def get_args(n_features):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    training_params ={
        'learning_rate': 0.1,
        'batch_size': 50,
        'epochs':500,
        'device': device,
        'n_features': n_features,
        'n_samples': 0.7 
    }

    testing_params = {
        'batch_size': 40,
        'n_samples': 0.3,
        'device': device
    }
    return training_params, testing_params

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
        last_model = torch.load('checkpoint/linear.pth', map_location=device)
        model.load_weight(last_model['params'])
        best_loss = last_model['loss']
        best_params = last_model['params']
    
    return model, best_params, best_loss

def save_model(best_params, best_loss):
    params = {
        'params': best_params,
        'loss': best_loss
    }
    torch.save(params, "checkpoint/linear.pth")

def split_train_test():
    database = FoodFeeling(has_bias=True)
    n_samples = database.n_samples
    training_params, testing_params = get_args(database.n_features)

    train_size = int(n_samples*training_params['n_samples'])
    test_size = n_samples - train_size

    train_set, test_set = random_split(database, [train_size, test_size], generator=torch.Generator().manual_seed(72))
    train_loader = DataLoader(train_set, batch_size=training_params['batch_size'])
    test_loader = DataLoader(test_set, batch_size=testing_params['batch_size'])

    return training_params, train_loader, test_loader

def train_and_evaluate():
    training_params ,train_loader, test_loader = split_train_test()
    epoch_error = train(train_loader, training_params)
    evaluate(training_params, test_loader)
    plot_loss(epoch_error[40:])

def plot_loss(epoch_loss):
    epoch_x_axis = [value[0] for value in epoch_loss]
    loss_y_axis = [value[1] for value in epoch_loss]

    plt.plot(epoch_x_axis, loss_y_axis)
    plt.xlabel('Epoch')
    plt.ylabel('Error of model')
    plt.show()
    return

def train(train_loader, training_params):
    
    epochs = training_params['epochs'] 
    device = training_params['device']
    model, best_params, best_loss = load_model(training_params=training_params)
    epoch_error =  list()
    # print(model.parameters(), best_params, best_loss)
    for epoch in range(epochs):
        total_loss = 0
        total_samples = 0
        model.learning_rate = lr_schedular(cur_epoch=epoch+1, lr=model.learning_rate, lr_decay=0.01, epoch_decay=250)
        for input_data, output_data in train_loader:

            model.zero_grad()
            output_data = output_data.view(1, -1).to(device)
            input_data = torch.transpose(input_data, 0, 1).to(device)
            y_hat = model(input_data)
            total_loss += model.MSELoss(y_hat, output_data)
            # print(model.parameters(), total_loss)

            total_samples += 1 
            model.train(input_data, output_data)
        if math.isnan(total_loss) == False and best_loss > total_loss/total_samples:
            best_loss = total_loss/total_samples
            best_params.copy_(model.parameters())
        
        cur_error = pow(total_loss/total_samples, 0.5)*5
        epoch_error.append([epoch, cur_error])
        print(f'{epoch} with loss mean {cur_error}')

    save_model(best_params=best_params, best_loss=best_loss)
    
    return epoch_error

def evaluate(training_params, test_loader):
    model = load_model(training_params)
    device = training_params['device']
    model, best_params, best_loss = load_model(training_params=training_params)
    # print(model.parameters(), best_params, best_loss)
    total_loss = 0
    total_samples = 0
    for input_data, output_data in test_loader:

        model.zero_grad()
        output_data = output_data.view(1, -1).to(device)
        input_data = torch.transpose(input_data, 0, 1).to(device)
        y_hat = model(input_data)
        # print(output_data*5)
        # print(y_hat*5)
        # print(output_data*5 - y_hat*5)
        total_loss += model.MSELoss(y_hat, output_data)

        total_samples += 1
    
    print(f'Mean Loss For testing data: {pow(total_loss/total_samples, 0.5)*5}')
if __name__ == '__main__':
    train_and_evaluate()