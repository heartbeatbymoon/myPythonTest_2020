# goodInfo={"id":1,"name":"shoes","product":"nike"}
# for key,value in goodInfo.items():
#     print(key,value)
# for key in goodInfo.keys():
#     print(key)

# item=["a","b","c"]
# i=0
# for index in item:
#     print("%d %s"%(i,index))
#     i+=1
# for key,value in enumerate(item) :
#     print("%d %s"%(key,value))

# a=[1,2]
# b=[3,4]
# print(a+b)
# c=[5,6]
# d=[7,8]
# print((c+d)*4)
#
# print(max("adfasdgasgdsfhsgfhsdfh"))
# print(min("adfasdgasgdsfhsgfhsdfh"))
#
# #
# def printInfo(a,b):
#     print("---------------")
#     print("%d   %d"%(a+b,a*b))
#     print("---------------")
#     return a+b,a*b
#
# key,value=printInfo(a=2,b=3)
# print(key,value)
sum = lambda a,b:a+b
print(sum(2,3))