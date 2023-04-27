# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Jagan a
RegisterNumber:  212221230037
*/
```

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("ex2data1.txt",delimiter = ',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y,np.log(1-h))) / X.shape[0]
    grad = np.dot(X.T, h - y) / X.shape[0]
    return J,grad
    
X_train = np.hstack((np.ones((X.shape[0],1)), X))
theta = np.array([0,0,0])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)), X))
theta = np.array([-24,0.2,0.2])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J
def gradient(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    grad = np.dot(X.T,h-y)/X.shape[0]
    return grad
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta  = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y),method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min, x_max = X[:, 0].min() - 1,X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1,X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max, 0.1),np.arange(y_min,y_max, 0.1))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot = np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
    X_train = np.hstack((np.ones((X.shape[0], 1)),X))
    prob = sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
    
np.mean(predict(res.x,X) == y)
```

## Output:

### Array Value of x
<img width="428" alt="pic1" src="https://user-images.githubusercontent.com/93427253/233859674-45ad7a9c-d9b4-4901-a492-77795ddc8a99.png">

### Array Value of y
<img width="308" alt="pic2" src="https://user-images.githubusercontent.com/93427253/233859691-2da3f1e0-411a-4a2a-9d9d-1b49c01a5e97.png">

### Exam 1- Score graph
<img width="444" alt="pic3" src="https://user-images.githubusercontent.com/93427253/233859703-9314a612-f936-4e38-94a9-7641e3651a2b.png">

### Sigmoid Function Graph
<img width="415" alt="pic4" src="https://user-images.githubusercontent.com/93427253/233859712-0f6629cc-9728-41be-94d6-e8f640657923.png">

### X_train_grad value
<img width="345" alt="pic5" src="https://user-images.githubusercontent.com/93427253/233859737-fde326a0-b68a-4364-8eeb-97cade466f82.png">

### Y_train_grad value
<img width="307" alt="pic6" src="https://user-images.githubusercontent.com/93427253/233859756-a00ca02e-0e81-45b4-ad01-6e961cd00b3a.png">

### Print res.x
<img width="356" alt="pic7" src="https://user-images.githubusercontent.com/93427253/233859766-98e987ed-0b56-45bd-b793-ed89183c4e10.png">

### Decision Boundary grapg for Exam Score
<img width="433" alt="pic8" src="https://user-images.githubusercontent.com/93427253/233859775-5a998730-65b5-4852-9f9c-0435e92b4b0a.png">

### Probability value
<img width="217" alt="pic9" src="https://user-images.githubusercontent.com/93427253/233859798-240dcbc8-389c-4b8d-8d0a-4b43a9acebdf.png">

### Prediction value of mean
<img width="260" alt="pic10" src="https://user-images.githubusercontent.com/93427253/233859803-a1dcbd59-37d4-42e8-aed0-4d5dfc91bc51.png">


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
