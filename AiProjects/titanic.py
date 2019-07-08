# encoding=utf-8
import pandas as pd

titanic = pd.read_csv("G:\\datas\\ai\\titanic_train.csv")
pd.set_option("display.max_columns", None)
# print(titanic.head())

# 如何查看是否有缺失值？？
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
# print(titanic.describe())

# 可以用改方判断是否有空值
print(titanic["Sex"].unique())

# 将性别量化
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

print(titanic["Sex"].unique())

print(titanic["Embarked"].unique())

titanic["Embarked"] = titanic["Embarked"].fillna("S")
print(titanic["Embarked"].unique())

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

print(titanic["Embarked"].unique())

# 使用线性回归进行机器学习
from sklearn.linear_model import LinearRegression
from  sklearn.cross_validation import KFold

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# alg = LinearRegression()
# kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
#
# predictions = []
# for train, test in kf:
#     train_predictors = (titanic[predictors].iloc[train, :])
#     train_target = titanic["Survived"].iloc[train]
#     alg.fit(train_predictors, train_target)
#     test_predictions = alg.predict(titanic[predictors].iloc[test, :])
#     predictions.append(test_predictions)
#
# import numpy as np
#
# # np.concatenate(predictions, axis=0)
#
# print(predictions)
# predictions[predictions > .5] = 1
# predictions[predictions <= .5] = 0
# accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
# print(accuracy)

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

print(scores)
print(scores.mean())

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

import re
def get_title(name):
    title_search= re.search("([A-Za-z]+)\.",name)
    if title_search:
        return title_search.group(1)
    return ""

titles = titanic["Name"].apply(get_title)
print(pd.value_counts(titles))

title_mapping ={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Dr":5,"Rev":6,"Major":7,"Col":7,"Mlle":8,"Mme":8,"Don":9,"Lady":10}
for k,v in title_mapping.items():
    titles[titles == k] = v

print(pd.value_counts(titles))
titanic["Title"] = titles