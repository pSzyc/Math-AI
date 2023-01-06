import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, output_size)
        self.Tanh = nn.Tanh()
        self.LeakyReLU = nn.LeakyReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.LeakyReLU(out)
        out = self.l2(out)
        out = self.Tanh(out)
        out = self.l3(out)
        return out