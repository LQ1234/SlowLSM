import numba as nb
import numpy as np
from numba.typed import List
@nb.njit()
def test(arr,tar):
    for i in range(len(tar)):
        for b in arr[i]:
            tar[i]+=b

def create_empty_nested_list(size):
    res=List()
    for itm in range(size):
        nested=List([0])
        nested.pop()
        res.append(nested)
    return(res)


arr=create_empty_nested_list(2)
arr[0].append(1)
arr[0].append(2)
tar=List([0,0])
print("star")
for i in range(10):
    test(arr,tar)
print(tar)
