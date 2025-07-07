import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets  import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


#load iris  dataset

iris  = load_iris()
X  = iris.data #feature 
y = iris.target # labels

#standardize data

scaler =  StandardScaler()
X = scaler.fit_transform(X)

#convert to pytorch tensors

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)


# Train-test spilt 

X_train, X_test, y_train, y_test  = train_test_split (X, y, test_size=0.3, random_state=42)


# Deffine fully Connected  Neural Network

class IrisNer(nn.Module):
    def __init__(self):
        super(IrisNer, self).__init__()
        self.fc1 = nn.Linear(4, 8) #input layer  4 feautures to 8 neurons
        self.relu = nn.ReLU()# Activation function

        self.fc2 = nn.Linear(8, 3) # 3output classes

    def forward(self, x):
        out = self.fc1(x)      # Pass input through first FC layer
        out = self.relu(out)   # Apply activation
        out = self.fc2(out)    # Pass through second FC layer (output layer)
        return out

    

# Instantiate model loss function and optimizer
model = IrisNer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


#Training loop
epochs = 100
for epoch in range(epochs):
    #forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    #backward  and Optimize 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# Testing
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = accuracy_score(y_test, predicted)
    print(f'\nTest Accuracy: {accuracy * 100:.2f}%')
