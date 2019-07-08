
#encoding=utf-8
a = (1,2,3)
b = (1,2,3)
c = zip(a,b)
d = list(c)
print(d)
print(type(c))
print(type(tuple(d)))