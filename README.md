# EE399. 
# introduction to machine learning. 
# HW2. 

## SVD and yalefaces. 
Author: Arvin Nusalim. 

Abstruct: given the data of 39 different faces with about 65 lighting scenes for each face, we tried to find the highly currelated and highly uncorrelated images between the given data. find the first largest 6 eigenvectors from the dot product of X and X transpose, find the 6 pricipal component direction using the SVD of matrix X, and compute the norm of difference of their abolute value. plot the 6 SVD modes after computing the percentage of variance.

## I. Introduction and Overview.   
This project's objective are to use the given data to find and plot the highly correlated images and highly uncorrelated images, compare the eigenvector and SVD modes, and use the first 6 SVD modes to compute the percentage of variance and plot to get the most dominant features. 

## II. Theoritical Background. 
In this section, we will provide a brief theoretical background on SVD. Singular Value Decomposition (SVD) is a method to break down a matrix into 3 parts that describe its most important features which are diagonal matrix ($S$) that measure how important the features and 2 orthogonal matrices ($U$ and $V^T$) that describe the direction of the features.   
![](SVD.png)     
SVD is usually used in the calculation of other matrix operations, such as matrix inverse, but also as a data reduction method in machine learning.

