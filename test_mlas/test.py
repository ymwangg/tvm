import tvm
import numpy as np
f = tvm._ffi.get_global_func("tvm.contrib.mlas.batch_matmul")
a = tvm.nd.array(np.random.random([16,16]).astype("float32"))
b = tvm.nd.array(np.random.random([16,16]).astype("float32"))
f(a,b)
f2 = tvm._ffi.get_global_func("tvm.contrib.mlas.gemm_packb_size")
res = f2(20, 20)
print(res)
f3 = tvm._ffi.get_global_func("tvm.contrib.mlas.gemm_packb")
f3(16,16,16,a,b)
