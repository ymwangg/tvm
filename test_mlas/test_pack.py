import numpy as np
from tvm import te
import tvm

def packb(K, N, B, transb=True):
    f = tvm._ffi.get_global_func("tvm.contrib.mlas.gemm_packb_size")
    packb_size = f(N, K)
    arr_size = int(packb_size / 4)
    return te.extern(
        (arr_size),
        [B],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.mlas.gemm_packb",
            N,
            K,
            K if transb else N,
            transb,
            ins[0],
            outs[0],
        ),
        name="PackedB",
    )

K, N = 300, 300
f = tvm._ffi.get_global_func("tvm.contrib.mlas.gemm_packb_size")
packb_size = f(N, K)
arr_size = int(packb_size / 4)
print(arr_size)

B = te.placeholder([K, N], dtype='float32')
packed = packb(K, N, B)
s = te.create_schedule([packed.op])
mod = tvm.build(s, [B, packed])
a = tvm.nd.array(np.random.random([K,N]).astype("float32"))
b = tvm.nd.array(np.zeros([arr_size]).astype("float32"))

mod(a, b)
print(a)
print(b)
