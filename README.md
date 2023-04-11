# EE399. 
## introduction to machine learning. 
## HW1. 

### curve fitting with line, parabola, and 19th degree polynomial. 
Author: Arvin Nusalim. 

Abstruct: Given the data, we tried to find the least-squares error and the parameters for that. Then we fix two of the parameters and sweep through values of the other two parameters to generate a 2D loss (error) landscape. do this for every combinations. some of the data are used to train the model and the rest are used to test the model. The model are used are line, parabola, and 19th degree polynomial.  

### I. Introduction and Overview.   
This project's objective are to use the given data to find the least-squares error and parameters, find the local minimum when 2 of the parameter are sweep through and 2 of them are fixed. least-squares error is a method used to determine the line of best fit for the data. We are using the least-squares error with a few function to find which one is the best fit for the data. 
   
### II. Theoritical Background. 
the function used are:  
$f(x) = A cos(Bx) + Cx + D   $  
$f(x) = Ax + B   $  
$f(x) = Ax^2 + Bx + C  $  
and $f(x) = 19^{th}$ degree polynomial   

using these function, find the least-squares error using the formula:   
$\sqrt{(1/n)\Sigma_{j=1}^n(f(x_j)-y_j)^2}$

### III. Algorithm Implementation and Development. 

### IV. Computational Results. 

### V. Summary and Conclusions. 
