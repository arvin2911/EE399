# EE399. 
# introduction to machine learning. 
# HW3. 

## Analysis on MNIST data
Author: Arvin Nusalim. 

Abstruct: Given a 70000 images of 28 X 28 pixel of digit from 0 to 9, perform an analysis of the MNIST data set. Perform SVD on the data set and On a 3D plot, project onto three selected V-modes (columns) colored by their digit label. After projecting the data set into PCA space, build 3 classifiers to identify individual digits. the 3 classifiers are Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), and Decision Tree.

## I. Introduction and Overview.   
This project's objective are to build 3 classifiers to be used on the given data. the 3 classifiers are Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), and Decision Tree. These classifiers will produce different accuracy in separating the 10 digits.

## II. Theoritical Background. 
In this section, we will provide a brief theoretical background on the 3 classifiers (LDA, SVM, and Decision Tree) which are used to clasify the MNIST data and Singular Value Decomposition (SVD). 

### 1. Linear Discriminant Analysis (LDA)
Linear Discriminant Analysis (LDA) is a supervised learning algorithm for classification. It find linear combination and feature that best separate the classes in the data. This can be done by maximizing the between-class variance and minimizing the within-class variance of the data. LDA is a method that assumes that the classes are normally distributed and the covariance matrices of the classes are equal.

### 2. Support Vector Machines (SVM)
Support Vector Machine (SVM) is a supervised learning algorithm that find a hyperplane in an N-dimensional space (N = number of features) that distinctly classifies the data points. firstly, it maps the data into a higher-dimentional feature space usng kernel function where the data will likely be more linearly separable in the higher dimension. Then, it find the best hyperplane that maximizes the margin between the classes in this higher-dimensional space. Although it is effective for high-dimentinal data, SVM can be computational expensive.

### 3. Decision Tree
Decision tree is a supervised machine learning alogrithm that uses a decision tree to classify new data points into several possible categories. This can be built by training on the labeled dataset. The decision tree is constructed by recursively partitioning the feature space into smaller regions based on the values of different features. When new data point is presented, it is classified by traversing the desition tree until it reach the leaf node, and then assign the label to the data point.

### SVD
Singular Value Decomposition (SVD) is a method to break down a matrix into 3 parts that describe its most important features which are diagonal matrix ($\Sigma$ or S) that measure how important the features and 2 orthogonal matrices ($U$ and $V^T$) that describe the direction of the features.   
![SVD.png](SVD.png)
