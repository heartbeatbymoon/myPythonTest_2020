# encoding=utf-8
import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from  sklearn.cross_validation import KFold

raw = pd.read_csv("G:\\datas\\ai\\data.csv")
print(raw.head(5))
print(raw.shape)

kobe = raw[pd.notnull(raw["shot_made_flag"])]
print(kobe)

alpha = 0.02  # 透明程度
plt.figure(figsize=(10, 10))

plt.subplot(121)
plt.scatter(kobe.loc_x, kobe.loc_y, color="blue", alpha=alpha)
plt.title("loc_x and loc_y")
plt.show()

plt.subplot(122)
plt.scatter(kobe.lon, kobe.lat, color="green", alpha=alpha)
plt.title("lat and lon")
plt.show()

raw["dist"] = np.sqrt(raw["loc_x"] ** 2 + raw["loc_y"] ** 2)
loc_x_zero = raw["loc_x"] == 0
raw["angle"] = np.array([0] * len(raw))
raw["angle"][~loc_x_zero] = np.arctan(raw["loc_y"][~loc_x_zero] / raw["loc_x"][~loc_x_zero])
raw["angle"][loc_x_zero] = np.pi/2

raw["remaining_time"] = raw["minuts_remaining"]*60+raw["seconds_remaining"]

# print(kobe.action_type.unique)
# print(kobe.combine)

print(kobe["season"].unique())
