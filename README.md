# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages required.
2.Read the dataset.
3.Define X and Y array.
4.Define a function for sigmoid, loss, gradient and predict and perform operations.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sarish Varshan V
RegisterNumber: 212223230196 
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv(""C:\Users\giri9\OneDrive\Documents\sa.csv")
dataset
dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values

Y

theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
    
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
    
y_pred = predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
## READ THE FILE AND DISPLAY
![image](https://github.com/sarishvarshan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152167665/2d2f57f2-07b0-4492-bd26-087e65e6ae77)
## Categorizing columns
![image](https://github.com/sarishvarshan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152167665/57e08967-0f66-4326-8345-979e9e64cd02)
## Labelling columns and displaying dataset
![image](https://github.com/sarishvarshan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152167665/4305329a-423c-4f2d-a706-7d1b6a291bcf)
## Display dependent variable
![image](https://github.com/sarishvarshan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152167665/66c1cada-19fd-484d-b184-a7188d24e4b0)
## Printing accuracy
![image](https://github.com/sarishvarshan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152167665/ac048119-52c6-48be-bf79-21f8f14caa34)
## Printing Y
![image](https://github.com/sarishvarshan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152167665/814294cb-2d0b-4a39-8cb4-bd8c75ecf6a1)
## Printing y_prednew
![image](https://github.com/sarishvarshan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152167665/774855ad-9280-4cd9-afb3-7f13e58fec73)







## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

