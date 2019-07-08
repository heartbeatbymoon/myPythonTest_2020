#encoding=utf-8
import pandas as pd
import numpy as np
# time = pd.date_range("20171001",periods=10,freq="3M")
# print(time)

new_time = pd.Series(np.random.rand(90),index=pd.date_range("2019-06-01",periods=90,freq="2D"))
print(new_time)
print(type(new_time))

# 通过时间切片将其取出
print(new_time["2019-06-01"])
new = new_time.resample("1M").sum()
print(new)

new1= new_time.resample("1M").asfreq()
print(new1)