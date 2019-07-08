#encoding=utf-8
import numpy as np

# def ultimate_answer(a):
#     result = np.zeros_like(a)
#     result.flat=42
#     return result
#
# ufunc = np.frompyfunc(ultimate_answer,1,1)
# print("The answer:"+ufunc(np.arange(4)))

a = np.arange(9)
print(a)
print(np.add.reduce(a))
print(np.add.accumulate(a))
print(np.add.reduceat(a,[0,5,2,7]))

