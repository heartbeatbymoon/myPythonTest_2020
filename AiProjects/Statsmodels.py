# encoding=utf-8
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# pd.set_option("display.max_columns",None)
data = pd.read_csv("G:\\datas\\ai\\Combined_News_DJIA.csv")
# print(data.head())
print(type(data))
train = data[data["Date"] < "2015-01-01"]
test = data[data["Date"] < "2014-12-31"]

example = train.iloc[3, 10]
print(example)

# 全部转换为小写
example2 = example.lower()
print(example2)

example3 = CountVectorizer().build_tokenizer()(example2)
print(example3)

df = pd.DataFrame([[x,example3.count(x)] for x in set(example3)],columns=["Word","Count"])
print(df)
