import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 4
num_classes = 3
hidden_size = 10
num_epochs = 100
learning_rate = 0.1

class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Prepare data
file = r"C:\Users\mwver\OneDrive\Documents\GitHub\Machine-Learning-and-Neural-Networks-Practice\iris_dataset\iris.csv"
df = pd.read_csv(file)

mappings = {
    "Setosa":0,
    "Versicolor":1,
    "Virginica":2
}

df["class"] = df["class"].apply(lambda x: mappings[x])

x = df.drop("class", axis=1).values
y = df["class"].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

x_train = torch.tensor(x_train, dtype=torch.float).to(device)
x_test = torch.tensor(x_test, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

losses = []

for epoch in range(num_epochs):

    # forward pass
    y_pred = model.forward(x_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    print(f'epoch: {epoch}, loss = {loss}')

    # backward pass
    optimizer.zero_grad()
    loss.backward()

    # update weights
    optimizer.step()

model.eval()
with torch.no_grad():

    y_pred = model(x_test)
    _, predicted_labels = torch.max(y_pred, 1)
    total_samples = y_test.size(0)
    correct_predictions = (predicted_labels == y_test).sum().item()
    accuracy = correct_predictions / total_samples * 100
    print('Accuracy: {:.2f}%'.format(accuracy))
    
    