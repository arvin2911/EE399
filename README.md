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

### Fit the Feedforward Neural Network
Firstly, we need to create a Feedforward Neural Network.
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 15)
        self.fc2 = nn.Linear(15, 6)
        self.fc3 = nn.Linear(6, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
Initialize the network and define the loss function and optimizer. we are using the nn.MSELoss to compute the mean square error or least square error.
```
model = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```
### define the parameter
```
# Define hyperparameters
dt = 0.01
T = 8
t = np.arange(0,T+dt,dt)
beta = 8/3
sigma = 10
# Given rho values to train and test
rho_values_train = [10, 28, 40]
rho_values_test = [17, 35]
```
### prepare the data to train
we need to insert the $\rho$ = 10, 28 and 40 to train the model. 
```
# Define the NN input and output
nn_input = np.zeros((100*(len(t)-1),3))
nn_output = np.zeros_like(nn_input)

# Create a list for input and output for the neural network (after data preparation)
nn_input_final = []
nn_output_final = []

# Data preparation
for rho in rho_values_train:
    
    # Define the Lorenz equation
    def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
        x, y, z = x_y_z
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    np.random.seed(123)
    x0 = -15 + 30 * np.random.random((100, 3))

    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t)
                      for x0_j in x0])
    
    for j in range(100):
        nn_input[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,:-1,:]
        nn_output[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,1:,:]
        
    # Convert numpy arrays to PyTorch tensors
    nn_input_tensor = torch.from_numpy(nn_input).float()
    nn_output_tensor = torch.from_numpy(nn_output).float()
    
    # Appending the tensors to a list
    nn_input_final.append(nn_input_tensor)
    nn_output_final.append(nn_output_tensor)
    

# Concatenate the neural network input and outputs from each rho values
nn_in = torch.cat(nn_input_final)
nn_out = torch.cat(nn_output_final)
```
### train the model
```
for epoch in range(30):
    optimizer.zero_grad()
    outputs = model(nn_in)
    loss = criterion(outputs, nn_out)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, loss={loss.item():.4f}")
```
### prepare the data to test
we need to insert the $\rho$ = 17 and 35 to test the model. 
```
nn_input_test_final = []
nn_output_test_final = []

# Test the network with given test rho values
for rho in rho_values_test:
    
    # Define the Lorenz equation
    def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
        x, y, z = x_y_z
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    np.random.seed(123)
    x0 = -15 + 30 * np.random.random((100, 3))

    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t)
                      for x0_j in x0])
    
    for j in range(100):
        nn_input[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,:-1,:]
        nn_output[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,1:,:]
        
    # Convert numpy arrays to PyTorch tensors
    nn_in_test_tensor = torch.from_numpy(nn_input).float()
    nn_out_test_tensor = torch.from_numpy(nn_output).float()
    
    # Appending the tensors to a list
    nn_input_test_final.append(nn_in_test_tensor)
    nn_output_test_final.append(nn_out_test_tensor)
    
# Concatenate the neural network input and outputs from each rho values
nn_in_test = torch.cat(nn_input_test_final)
nn_out_test = torch.cat(nn_output_test_final)
```
### test the model
```
# Test the network
with torch.no_grad():
    outputs = model(nn_in_test)
    compute_mse = criterion(outputs, nn_out_test)

    print('Least squares error of test data: {}'.format(compute_mse.item()))
```
### fit the Long Short term Memory, train, and test the model

### fit the Recurrent Neural Network, train, and test the model

### fit the Echo State Network, train, and test the model


## IV. Computational Results.

## V. Summary and Conclusions.

