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
### fit a Feedforward Neural Network into a 1D data
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
### fit a Feedforward Neural Network into a 2D data


## IV. Computational Results.

## V. Summary and Conclusions.
