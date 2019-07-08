#encoding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
admissions = pd.read_csv("G:\\datas\\ai\\admissions.csv")
print(admissions.head(5))
plt.scatter(admissions["gpa"],admissions["admit"])
plt.show()

import numpy as np

def logit(x):
    return np.exp(x)/(1+np.exp(x))

x = np.linspace(-6,6,50,dtype=float)
y = logit(x)

plt.plot(x,y)
plt.ylabel("Probability")
plt.show()

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(admissions[["gpa"]],admissions["admit"])
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(admissions[["gpa"]],admissions["admit"])
pred_probs = logistic_model.predict_proba(admissions[["gpa"]])
plt.scatter(admissions["gpa"],pred_probs[:,1])
plt.show()