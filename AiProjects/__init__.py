#encoding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
columns = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin","car name"]
cars = pd .read_table("G:\\datas\\ai\\auto-mpg.data",delim_whitespace=True,names=columns)
print(cars.head(5))

fig = plt.figure()    #创建一个画图板
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
cars.plot("weight","mpg",kind = "scatter",ax = ax1)
cars.plot("acceleration","mpg",kind = "scatter",ax = ax2)
#plt.show()


import sklearn
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
res = lr.fit(cars[["weight"]],cars["mpg"])
print(res)

predictions = lr.predict(cars[["weight"]])
print(predictions[0:5])
print(cars["mpg"][0:5])

plt.scatter(cars["weight"],cars["mpg"],c="red")
plt.scatter(cars["weight"],predictions,c="blue")
plt.show()

from sklearn.metrics import  mean_squared_error
mse = mean_squared_error(cars["mpg"],predictions)
print(mse)

