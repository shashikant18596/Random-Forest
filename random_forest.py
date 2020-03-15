
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv("C:\\Users\shashikant\Desktop\polynomial_regression\polynomial.csv")
df

model = RandomForestRegressor(n_estimators=800,random_state=0)

x = df[['level']].values
x

y = df[['salary']].values
y

model.fit(x,y)

model.predict([[6]])


x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
x_grid


plt.title('Random Forest Algorithm')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.scatter(x,y,color='r')
plt.plot(x_grid,model.predict(x_grid),color='b')

