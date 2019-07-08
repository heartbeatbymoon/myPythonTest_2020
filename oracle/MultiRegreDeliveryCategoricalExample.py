#encoding=utf-8
from numpy import genfromtxt
import numpy as np
from sklearn import  datasets,linear_model
# r 的含义是省去路径中的转义，更安全
dataPath = r"G:\datas\MultiLinearRegression\Delivery_Dummy.csv";
deliveryData = genfromtxt(dataPath,delimiter=",")
print(deliveryData)
print(dataPath)


# :代表所有行   -1代表代表到倒数第一列但是不包括倒数第一列
X = deliveryData[:,:-1]
Y = deliveryData[:,-1]
print("X:")
print(X)
print("Y:")
print(Y)

regr = linear_model.LinearRegression()
regr.fit(X,Y)

print("coefficients")
# 属性值
print(regr.coef_)
print("intercept:")
print(regr.intercept_)

xPred = [102,6]
xPred= np.array(xPred).reshape(1,-1)
yPred = regr.predict(xPred)
print("Predict y:")
print(yPred)



