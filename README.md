# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: V.Shriyha
RegisterNumber: 212224230267
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions - y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_scaled=scaler.fit_transform(X1)
Y1_scaled=scaler.fit_transform(y)
print(X1_scaled)
print(Y1_scaled)
theta=linear_regression(X1_scaled,Y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

```

## Output:
![Screenshot 2025-04-07 152428](https://github.com/user-attachments/assets/302848f1-70c7-4d46-9c25-1105ffede503)

![Screenshot 2025-04-07 151935](https://github.com/user-attachments/assets/beff308f-386a-4c19-b4aa-31c5da3fa25a)

![Screenshot 2025-04-07 151944](https://github.com/user-attachments/assets/e12ba9ef-32a6-4521-a61a-722160e657e7)

![Screenshot 2025-04-07 151958](https://github.com/user-attachments/assets/bd777add-0128-48ab-a9ff-fd222e8665e2)

![Screenshot 2025-04-07 152005](https://github.com/user-attachments/assets/7fac404a-043e-40ab-a336-1d0c2421fd32)

![Screenshot 2025-04-07 152012](https://github.com/user-attachments/assets/1693a74c-ac13-43ef-be30-ca1900266b51)

![Screenshot 2025-04-07 152018](https://github.com/user-attachments/assets/acf0f713-9919-4c36-8184-c1f5534cdae1)

![Screenshot 2025-04-07 152026](https://github.com/user-attachments/assets/c2de5754-7e1f-4425-8d95-75768039ed40)

![Screenshot 2025-04-07 152044](https://github.com/user-attachments/assets/c3e0daf9-e91c-431f-bd49-92375304e104)

![Screenshot 2025-04-07 152051](https://github.com/user-attachments/assets/0eb70fbe-4806-4081-bf28-806225d24e32)

![Screenshot 2025-04-07 152058](https://github.com/user-attachments/assets/fc0935a0-7726-4955-b50f-f618c29f3623)

![Screenshot 2025-04-07 152105](https://github.com/user-attachments/assets/ed612162-110a-4925-8921-0ba52f287f77)

![Screenshot 2025-04-07 152127](https://github.com/user-attachments/assets/b35568e4-96d2-48de-b86a-58a50b7211ea)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
