import time
import numpy as np
import copy
import ctypes


def binary(inputs,threshold=0.5):
    inputs_ptr = ctypes.cast(inputs.ctypes.data,ctypes.POINTER(ctypes.c_double))
    size = np.array(inputs.shape,dtype=int)
    size_ptr = ctypes.cast(size.ctypes.data,ctypes.POINTER(ctypes.c_int))
    output = np.zeros(inputs.shape)
    output_ptr = ctypes.cast(output.ctypes.data,ctypes.POINTER(ctypes.c_double))
    Lib.BinaryThreshold(inputs_ptr,output_ptr,size_ptr,ctypes.c_double(threshold))
    return copy.deepcopy(output)


a = np.random.random([291, 512, 512])
print(a[0])
begin = time.time()
a[a >= 0.5] = 1
a[a < 0.5] = 0
print(time.time()-begin)
print(a[0])

Lib = ctypes.cdll.LoadLibrary("C:\\Users\\Administrator\\Desktop\\Filter\\Lib.so")
print("Load Lib")
a = np.random.random([291,512,512])
begin = time.time()
a_out = binary(copy.deepcopy(a))

print(a[0])
print(a_out[0])
print(time.time()-begin)