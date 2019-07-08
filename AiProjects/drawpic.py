# encoding=utf-8
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False

# numbers = list(range(1,11))
# #np.array()将列表转换为存储单一数据类型的多维数组
# x = ['2015', '2016', '2017', '2018', '2019E', '2020E']
# y = [1684, 1971, 2307, 2700, 4285, 6800]
# plt.bar(x,y,width=0.5,align='center')
# plt.title('全球人工智能市场规模（2015-2020）',fontsize=16)
# plt.xlabel('年份',fontsize=13)
# plt.ylabel('亿人民币',fontsize=13)
# plt.tick_params(axis='both',labelsize=13)
#
# # 这个是为了添加柱状图上面的数字
# for a,b in zip(x,y):
#     plt.text(a,b,'%.0f'%b,ha = 'center',va = 'bottom',fontsize = 12)
# plt.show()


numbers = list(range(1,11))
#np.array()将列表转换为存储单一数据类型的多维数组
x = ['2015', '2016', '2017', '2018', '2019E', '2020E']
y = [112.4, 141.9, 216.9, 339, 500, 710]
plt.bar(x,y,width=0.5,align='center')
plt.title('中国人工智能市场规模及预测（2015-2020）',fontsize=16)
plt.xlabel('年份',fontsize=13)
plt.ylabel('亿人民币',fontsize=13)
plt.tick_params(axis='both',labelsize=13)

# 这个是为了添加柱状图上面的数字
for a,b in zip(x,y):
    plt.text(a,b,'%.0f'%b,ha = 'center',va = 'bottom',fontsize = 12)
plt.show()

