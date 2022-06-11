from Linear.Linear_GPU import Linear
from utils.utils import FoodFeeling
import torch
from torch.utils.data import DataLoader

input_dim = 3
learning_rate = 0.01
batch_size = 50 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
data = FoodFeeling()
model = Linear(input_dim=input_dim, lr=learning_rate, batch_size=batch_size, device=device).to(device)
data_loader = DataLoader(data, batch_size=batch_size)
print('DEVICE: ', device)
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
