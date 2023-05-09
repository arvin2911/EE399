# EE399. 
# introduction to machine learning. 
# HW4. 

## Analysis on 3 layer Feedforward Neural Network
Author: Arvin Nusalim. 

Abstruct: Given a dataset, use a certain amount of data as a training data to fit in a 3 layer Feedforward Neural Network to compute the least square error model and use the model to the rest of the data. Given a MNIST dataset, perform a PCA and fit it in a Feedforward Neural Network, then compare it with LSTM, SVM, and decision tree.

## I. Introduction and Overview.
This project's objective are to build a 3 layer Feedforward Neural Network (FFNN) and fit it to the given data. First, we will use a 3-layer Feedforward Neural Network (FNN) to perform a least square error regression on a portion of the dataset used as training data. The resulting model will then be used to predict the labels of the rest of the dataset. Next, we will perform Principal Component Analysis (PCA) on the MNIST dataset to reduce its dimensionality and train the FFNN model using the reduced dataset. We will compare the performance of this FFNN model with other machine learning algorithms such as LSTM, SVM, and decision tree.

## II. Theoritical Background.
In this section, we will provide a brief theoretical background on Feedforward Neural Network (FFNN), Long Short-term Memory (LSTM), Support Vector Machines (SVM), and decision tree.

### 1. Feedforward Neural Network (FFNN)
A Feedforward Neural Network (FFNN) is a type of machine learning model that takes input and produces output in a single direction, with multiple layers of interconnected neurons in between. It's commonly used for tasks like predicting a label for an input. During training, the network adjusts its weights and biases to better match the desired output.

### 2. Long Short-term Memory (LSTM)
Long Short-Term Memory (LSTM) is a type of neural network that is good at handling sequential data like text or speech. It has a special memory cell that can store information over time and three different types of gates to control the flow of information. LSTM networks can remember or forget information as needed and are commonly used for natural language processing and speech recognition.

### 3.  Support Vector Machine (SVM)
Support Vector Machine (SVM) is a supervised learning algorithm that find a hyperplane in an N-dimensional space (N = number of features) that distinctly classifies the data points. firstly, it maps the data into a higher-dimentional feature space usng kernel function where the data will likely be more linearly separable in the higher dimension. Then, it find the best hyperplane that maximizes the margin between the classes in this higher-dimensional space. Although it is effective for high-dimentinal data, SVM can be computational expensive.

### 4. Decision tree
Decision tree is a supervised machine learning alogrithm that uses a decision tree to classify new data points into several possible categories. This can be built by training on the labeled dataset. The decision tree is constructed by recursively partitioning the feature space into smaller regions based on the values of different features. When new data point is presented, it is classified by traversing the desition tree until it reach the leaf node, and then assign the label to the data point.

## III. Algorithm Implementation and Development.
### Necessary import
the necessary import for this project are
```
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import scipy.optimize as opt
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
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
Initialize the network and define the loss function and optimizer.
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
## IV. Computational Results.

## V. Summary and Conclusions.
