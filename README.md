# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. import required libraries (NumPy, Pandas, Matplotlib).
 2. Load the dataset from Startup.csv,Select input (R&D Spend) and output (Profit),Convert data into NumPy arrays.
 3. Normalize the input data,Initialize slope m = 0 and intercept b = 0,Set learning rate and number of iterations (epochs).
 4. Repeat for many iterations:
   -->  Predict values using y = mx + b
   -->  Find error between actual and predicted values
   -->  Calculate changes for m and b
   -->  Update m and b to reduce error
   -->  After training, get final values of m and b
   --> Use final equation to predict profit
   -->  Plot actual data and best-fit line on graph
  

## Program:
/*
Program to implement the linear regression using gradient descent.

Developed by:Jones Benedict A P

RegisterNumber:212224040142 

*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load the Data
data=pd.read_csv("50_Startups.csv")
x=data["R&D Spend"].values
y=data["Profit"].values

#Feature Scaling
x_mean=np.mean(x)
x_std=np.std(x)
x=(x-x_mean)/x_std

#Parameters
w=0.0
b=0.0
alpha=0.01
epochs=100
n=len(x)

losses=[]

#Gradient Descent
for i in range(epochs):
    y_hat=w*x+b
    loss=np.mean((y_hat-y)**2)
    losses.append(loss)
    
    dw=(2/n)*np.sum((y_hat-y)*x)
    db=(2/n)*np.sum(y_hat-y)
    
    w-=alpha*dw
    b-=alpha*db

#Plot
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss(MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1,2,2)
plt.scatter(x,y)
x_sorted=np.argsort(x)
plt.plot(x[x_sorted],(w*x+b)[x_sorted],color="red")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression Fit")

plt.tight_layout()
plt.show()

print(f"Final weight (w): {w}")
print(f"Final bias (b): {b}")
```

## Output:
<img width="891" height="406" alt="image" src="https://github.com/user-attachments/assets/c0559405-7be8-4407-adae-f62d1d4bba92" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
