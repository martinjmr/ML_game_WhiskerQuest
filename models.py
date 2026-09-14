import torch 
from torch import nn            

class DenseModel(nn.Module):
    def __init__(self, n_state, n_action):
        super(DenseModel, self).__init__()
        
        self.fc1 = nn.Linear(n_state * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_action)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = torch.flatten(x, start_dim=1)
        
        
        
        x = x.view(x.size(0), -1)
        
        
        x = self.activation(self.fc1(x))
        
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
    

class CNNModel(nn.Module):
    def __init__(self, n_state, n_action):
        super(CNNModel, self).__init__()
        self.x_shape = (5, 5, n_state)
        self.conv1 = nn.Conv2d(n_state, 8, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=1)
        self.fc = nn.Linear(16*3*3, n_action)
        self.activation = nn.ReLU()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        


