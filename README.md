# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. NumPy, pandas, and StandardScaler are imported for calculations, data handling, and feature scaling.
2. Define Linear Regression: A function using gradient descent is created to update weights and minimize prediction error.
3. Load & Preprocess Data: The dataset is loaded, features and target are separated, and both are standardized.
4. Train the Model: Gradient descent is applied to find the best parameters.
5. Make Predictions: New data is scaled and passed to the model, then converted back to original scale.
6. Print Output: The predicted value is displayed.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Karthic U
RegisterNumber: 212224040151 
```

## Dataset:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset
```


## Output:
<img width="1420" height="510" alt="image" src="https://github.com/user-attachments/assets/e56b31e7-bf2b-4461-9aa7-9cf322f82478" />



## Types:
```py
if 'sl_no' in dataset.columns:
    dataset=dataset.drop("sl_no",axis=1)
if 'salary' in dataset.columns:
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

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```

## Output:
<img width="1148" height="506" alt="image" src="https://github.com/user-attachments/assets/d49544a7-3676-4db0-97c3-d074f58cf52f" />


## Accuracy:
```py
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

Y

theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

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
print("Accuracy:",accuracy)
print(Y)
```

## Output:
<img width="787" height="157" alt="image" src="https://github.com/user-attachments/assets/82967533-a90e-4384-8bc6-cff39ff53a0f" />

## Predicted values:
```py
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)


xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
<img width="88" height="31" alt="image" src="https://github.com/user-attachments/assets/093edcd2-0ad9-465e-8757-247750246e7a" />

<img width="91" height="33" alt="image" src="https://github.com/user-attachments/assets/e7c23f41-02cd-4ac7-8a17-6cba744a76f6" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

