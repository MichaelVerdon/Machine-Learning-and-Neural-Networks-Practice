import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file = r"C:\Users\mwver\OneDrive\Documents\GitHub\Machine-Learning-and-Neural-Networks-Practice\breast_cancer\breast-cancer-wisconsin.data"
df = pd.read_csv(file, header=None, na_values="?")
df.dropna(inplace=True)

headers = ["ID", "Clump Thickness", "Uniformity of Cell Size",
           "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
           "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]

df.columns = headers

vars = ["Clump Thickness", "Uniformity of Cell Size",
        "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
        "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]

n_inputs, n_features = df[vars[:-1]].shape
print(n_features, n_inputs)

learning_rate = 0.1
epochs = 100
hidden_size = 20
num_classes = 1

class NeuralNet(nn.Module):

    def __init__(self, n_features, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(n_features, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = nn.ReLU()(self.layer1(x))
        out = nn.Sigmoid()(self.layer2(out))
        return out
    
model = NeuralNet(n_features, hidden_size, num_classes)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

x = df[vars[:-1]].values
y = df["Class"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(np.shape(y_train), np.shape(y_test))

x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

epochs = 100

for epoch in range(epochs):

    # Forward Pass
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    print(f'epoch: {epoch}, loss = {loss.item()}')

    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model.eval()
