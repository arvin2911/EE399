# EE399. 
# introduction to machine learning. 
# HW5. 

## FFNN, LSTM, RNN, ESN for future state prediction on Lorenz equations.
Author: Arvin Nusalim. 

Abstruct: using Feedforward Neural Network, Long Short-term Memory, Recurrent Neural Network, and Echo State Network on Lorenz equations for future state prediction. Train the model using training data from Lorenz equations for $\rho$ = 10, 28 and 40 and test the data for $\rho$ = 17 and 35.

## I. Introduction and Overview.   
This project's objective are to built Feedforward Neural Network (FFNN) and fit it to the lorenz equations. First, train the model and perform least square error on lorenz equation. The resulting model then be used to predict the future state of lorenz equation. we will then compare the result of FFNN with the result from Long Short-term Memory (LSTM), Recurrent Neural Network (RNN), and Echo State Network (ESN).
   
## II. Theoritical Background.
In this section, we will provide a brief theoretical background on Feedforward Neural Network (FFNN), Long Short-term Memory (LSTM), Recurrent Neural Network (RNN), and Echo State Network (ESN).

### 1. Feedforward Neural Network (FFNN)
A Feedforward Neural Network (FFNN) is a type of machine learning model that takes input and produces output in a single direction, with multiple layers of interconnected neurons in between. It's commonly used for tasks like predicting a label for an input. During training, the network adjusts its weights and biases to better match the desired output.

### 2. Long Short-term Memory (LSTM)
Long Short-Term Memory (LSTM) is a type of neural network that is good at handling sequential data like text or speech. It has a special memory cell that can store information over time and three different types of gates to control the flow of information. LSTM networks can remember or forget information as needed and are commonly used for natural language processing and speech recognition.

### 3.  Recurrent Neural Network (RNN)
Recurrent Neural Network (RNN) is a type of neural network commonly used for tasks involving sequences of data. It can remember previous inputs by using a hidden state, allowing it to learn patterns or dependencies in sequential data. RNNs are useful for tasks like language processing and time series prediction. However, they have difficulty capturing long-term dependencies.

### 4. Echo State Network (ESN)
An Echo State Network (ESN) is a type of recurrent neural network (RNN) that is known for its simplicity and effectiveness in handling sequential data. It uses randomly initialized connections that remain fixed during training, while only the connections between input and output layers are learned. This fixed random initialization creates a reservoir of hidden states, which act as memory for processing sequential data. ESNs are computationally efficient, robust to gradient problems, and have been successfully applied to various tasks such as time series prediction and speech recognition.

## III. Algorithm Implementation and Development.
### Necessary import
the necessary import for this project are
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import torch
import torch.nn as nn
import torch.optim as optim
```
### Fit a Feedforward Neural Network into a 1D data
Firstly, we need to create a Feedforward Neural Network.
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
Initialize the network and define the loss function and optimizer. we are using the nn.MSELoss to compute the mean square error or least square error.
```
net = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
```

Given the data:
```
X = np.arange(0, 31).reshape(-1, 1)
Y=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53]).reshape(-1, 1)
```
Using the first 20 data points as training data, fit the neural network and compute the least-square error for each of the training points. Then use the model on the test data which are the remaining data points.
```
# Prepare the data
Y_train = Y[:20].astype(np.float32)
X_train = X[:20].astype(np.float32)
Y_test = Y[20:].astype(np.float32)
X_test = X[20:].astype(np.float32)
X_train_tensor = torch.from_numpy(X_train)
Y_train_tensor = torch.from_numpy(Y_train)
X_test_tensor = torch.from_numpy(X_test)
Y_test_tensor = torch.from_numpy(Y_test)

# Train the neural network
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()   # zero the gradients
    output = net(X_train_tensor)  # forward pass
    loss = criterion(output, Y_train_tensor)  # compute the loss
    loss.backward()  # backward pass
    optimizer.step()  # update the weights
    
    # Print the loss every epoch
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# MSE on the test data
with torch.no_grad():
    output = net(X_test_tensor)
    test_mse = criterion(output, Y_test_tensor)
    print('MSE of test data: {:.4f}'.format(test_mse.item()))
```
Do the same thing using the first 10 and last 10 as the training data to fit the neural network and compute the least-square error for each of the training points. Then use the model on the test data which are the remaining data points.
```
# Prepare the data
Y_train = np.concatenate((Y[:10],Y[21:])).astype(np.float32)
X_train = np.concatenate((X[:10],X[21:])).astype(np.float32)
Y_test = Y[10:21].astype(np.float32)
X_test = X[10:21].astype(np.float32)
X_train_tensor = torch.from_numpy(X_train)
Y_train_tensor = torch.from_numpy(Y_train)
X_test_tensor = torch.from_numpy(X_test)
Y_test_tensor = torch.from_numpy(Y_test)
```
we can use the same code to train the neural network and compute the least-square error.

### Fit a Feedforward Neural Network into a 2D data
Firstly, we need to create a neural network.
```
# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 100)
        self.fc2 = nn.Linear(100, 500)
        self.fc3 = nn.Linear(500, 100)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
We can the grab the MNIST data, apply transformations, and reshape the 2D images into 1D.
```
# Load the MNIST dataset and apply transformations
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Get the pixel values for all images in the dataset
X_train = train_dataset.data.float()
y_train = train_dataset.targets
X_test = test_dataset.data.float()
y_test = test_dataset.targets

# Reshape the 2D images into 1D vectors
X_train = X_train.view(X_train.size(0), -1)
X_test = X_test.view(X_test.size(0), -1)
```

transform into 20 dimension using PCA and turn the data back into tensor type.
```
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

# Convert numpy arrays back to tensors
X_train_pca_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
X_test_pca_tensor = torch.tensor(X_test_pca, dtype=torch.float32)

# Create dataloaders for the training and test data
train_data = torch.utils.data.TensorDataset(X_train_pca_tensor, y_train)
test_data = torch.utils.data.TensorDataset(X_test_pca_tensor, y_test)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
```
Initialize the network and define the loss function and optimizer. In PyTorch, nn.CrossEntropyLoss is commonly used for classification problems where the model needs to predict a discrete label for each input.
```
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
```
We then need to train the network and test it.
```
# Train the network
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
            
# Test the network
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
```
### LSTM on MNIST
to compute the accuracy for LSTM on MNIST, we can use the code shown below.
```
# Define the LSTM network architecture
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 28
sequence_length = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 64
num_epochs = 5
learning_rate = 0.01

# Load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMNet(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
```
### SVM and decision tree on MNIST
Firstly, we need to load the data and convert into numpy array.
```
# Load the MNIST dataset from OpenML
mnist = fetch_openml('mnist_784')

# Convert the data and labels to numpy arrays
X = mnist.data.astype('float32') / 255.0    
y = mnist.target.astype('int32')
```
Perform PCA to the data.
```
# Perform PCA to reduce the dimensionality of the data
pca = PCA(n_components=20)
X_pca_train = pca.fit_transform(X)
```
Split the data into training data and testing data.
```
# Split the whole dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca_train, y, test_size=0.2, random_state=42)
```
Compute the percent accuracy for SVM by using 
```
# Train an SVM classifier on the training set
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
```
and compute the percent accuracy for decision tree by using
```
# Train a decision tree classifier on the training set
clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
```
## IV. Computational Results.
### Feedforward Neural Network onto 1D data.
After using the 3 layer Feedforward Neural Network we build and train using the training data, we will use the model to compute the least square error or least square error.
Using the first 20 data as the training data and the rest as the test data, the Mean Square Error for the test data is 147.0718 while by using the first 10 data and last 10 data as the training data and the rest as the test data, the Mean Square Error for the test data is 30.5956. This shows that using the first 10 data and last 10 data as the training data and the rest as the test data will give a lower Mean Square Error compared with using the first 20 data as the training data and the rest as the test data. If we compare these 2 model with the model from the first project (HW1), the first one give a better result than this.

### Feedforward Neural Network onto 2D data.
After using the 3 layer Feedforward Neural Network we build and train using the training data, we will use the model to compute the accuracy of the Feedforward Neural Network on MNIST data was got 95.29%. We also compute the percent accuracy of the LSTM on the MNIST test images and got 97.42%. The same goes for SVM and decision tree which we got 90.61% and 79.34% consecutively.

## V. Summary and Conclusions.
To conclude, for computing Mean Square Error on 1D data on a small dataset, its better to use Linear or Parabola compared to using Feedforward Neural Network. While for computing the accuracy on 2D data, the best classifier to use are LSTM because LSTM gives the highest percent accuracy which is 97.42%. After LSTM the accuracy rank are FFNN, SVM, and decision tree consecutively.
