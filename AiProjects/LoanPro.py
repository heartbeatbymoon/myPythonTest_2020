# encoding=utf-8
import numpy as np
import pandas as pd

# 数据预处理

# 跳过行读取csv
loan_data = pd.read_csv("G:\\datas\\ai\\LoanStats3a.csv", skiprows=1)
# print(loan_data.dtypes.value_counts())
print(loan_data.shape)
# 去掉空值(列，如果空值大于行数的一半，过滤掉)
loan_data = loan_data.dropna(thresh=len(loan_data) / 2, axis=1)
print(loan_data.shape)
loan_data = loan_data.drop(
    ["desc", "id", "member_id", "funded_amnt", "funded_amnt_inv", "grade", "sub_grade", "emp_title", "issue_d",
     "zip_code",
     "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", \
     "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee",
     "last_pymnt_d", "last_pymnt_amnt"], axis=1)
print(loan_data.shape)

# 找出分类依据（最终放贷结果）
print(loan_data["loan_status"].value_counts())
# Fully Paid                                             33902
# Charged Off                                             5658
# Does not meet the credit policy. Status:Fully Paid      1988
# 如何删选出有效项目？
loan_data = loan_data[(loan_data["loan_status"] == "Fully Paid") | (loan_data["loan_status"] == "Charged Off")]
print(loan_data.shape)

status_replace = {
    "loan_status": {
        "Fully Paid": 0,
        "Charged Off": 1
    }
}
loan_data = loan_data.replace(status_replace)
print(loan_data["loan_status"].value_counts())

# 过滤掉参数中仅仅含有同样属性的列
print(loan_data.columns)
columns = loan_data.columns
drop_cols = []
for col in columns:
    drop = loan_data[col].dropna().unique()
    if len(drop) == 1:
        drop_cols.append(col)

print(drop_cols)
loan_data = loan_data.drop(drop_cols, axis=1)
print(loan_data.shape)

# loan_data.to_csv("G:\\datas\\ai\\my_washedLoan.csv")
print("++++++++++++++++++清洗后数据处理++++++++++++++++++")

loans = pd.read_csv("G:\\datas\\ai\\my_washedLoan.csv")
print(loans.shape)
null_counts = loans.isnull().sum()
print(null_counts)

loans = loans.drop("pub_rec_bankruptcies", axis=1)
loans = loans.dropna(axis=0)
print(loans.isnull().sum())

print(loans.dtypes.value_counts())

object_col = loans.select_dtypes(include="object")
print(object_col.iloc[0])

cols = ['home_ownership', 'verification_status', 'emp_length', 'term', 'addr_state']
for c in cols:
    print(loans[c].value_counts())

mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}

loans = loans.drop(["url", "last_credit_pull_d", "earliest_cr_line", "addr_state", "title"], axis=1)
loans["int_rate"] = loans["int_rate"].str.rstrip("%").astype("float")
loans["revol_util"] = loans["revol_util"].str.rstrip("%").astype("float")
loans = loans.replace(mapping_dict)

cat_col = ["home_ownership", "verification_status", "emp_length", "purpose", "term"]
dummy_df = pd.get_dummies(loans[cat_col])
loans = pd.concat([loans, dummy_df], axis=1)
print(loans.shape)
loans = loans.drop(cat_col, axis=1)
loans = loans.drop("pymnt_plan", axis=1)

print(loans.shape)
print(loans.dtypes.value_counts)
# loans.to_csv("G:\\datas\\ai\\_washedLoan.csv")


loans = pd.read_csv("G:\\datas\\ai\\cleaned_loans2007.csv")
print(loans.info())

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(class_weight="balanced")
cols = loans.columns
print(cols)

train_cols = cols.drop("loan_status")
features = loans[train_cols]
target = loans["loan_status"]
lr.fit(features, target)
predictions = lr.predict(features)
print(predictions)

from sklearn.cross_validation import cross_val_predict, KFold

kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)

# False positives
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

tpr = tp/float((tp+fn))
fpr = fp/float((fp+tn))

print(tpr)
print(fpr)
print(predictions[:20])

