from math_model import NeuralNet
from prepare_data import converter, MathDataset

import numpy as np
import random
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


df = pd.read_csv('train_data.txt')
df['ExpressionVector'] = df['Expression'].apply(converter)
X_train = torch.tensor(df['ExpressionVector'], dtype = torch.float32)
Y_train = torch.tensor(df['Value'], dtype = torch.float32)
dataset = MathDataset(X_train, Y_train)
train_loader = DataLoader(dataset= dataset,
                          batch_size = 2000,
                          shuffle=True,
                          num_workers=0)

df = pd.read_csv('test_data.txt')
df['ExpressionVector'] = df['Expression'].apply(converter)
X_test = torch.tensor(df['ExpressionVector'], dtype = torch.float32)
Y_test = torch.tensor(df['Value'], dtype = torch.float32)
dataset = MathDataset(X_train, Y_train)
test_loader = DataLoader(dataset= dataset,
                          batch_size=1000,
                          shuffle=True,
                          num_workers=0)

model = NeuralNet(input_size=6, hidden_size=100, output_size=1)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []
losses_on_test = []
num_epochs = 20000

for epoch in range(num_epochs):
    for (expressions, values) in train_loader:    
        outputs = model(expressions)
        loss = loss_function(outputs.reshape(-1), values)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    for (expressions, values) in train_loader:
        with torch.no_grad():
            outputs = model(expressions)
            loss_on_test = loss_function(outputs.reshape(-1), values)
            
        
        
    if (epoch) % 500 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Loss on test: {loss_on_test.item():.4f}')
        losses.append(loss.item())
        losses_on_test.append(loss_on_test.item())

# saving model

torch.save(model.state_dict(), f'learned{num_epochs}.pth')
print(losses)

fig, ax = plt.subplots()
plt.plot([i + 1 for i in range(len(losses))], losses)
plt.plot([i + 1 for i in range(len(losses))], losses_on_test)
plt.savefig(f"leraning: {num_epochs}.pdf")

print(len(losses))