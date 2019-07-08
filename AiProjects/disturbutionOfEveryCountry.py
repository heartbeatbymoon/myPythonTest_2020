#encoding=utf-8
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False

country = ["美国","中国","英国","加拿大","印度","以色列","法国","德国","瑞典","西班牙","日本","瑞士","荷兰","波兰"\
           ,"澳大利亚","意大利","爱尔兰","新加坡","韩国","俄国"]
num = [2039,1040,392,287,152,122,121,111,56,53,40,40,40,33,31,30,25,25,25,17]

plt.figure()
plt.bar(country,num)
plt.xlabel("国家",fontsize=16)
plt.ylabel("企业数量",fontsize=16)
plt.xticks(rotation=45)

for x,y in zip(country,num):
    plt.text(x,y,'%.0f'%y,ha = 'center',va = 'bottom',fontsize=12)
plt.show()
# print(plt.style.available)