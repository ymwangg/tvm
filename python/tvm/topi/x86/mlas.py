from tvm import te
import tvm
from tvm.topi.utils import get_const_float, get_const_tuple


def mlas_packb(K, N, B, transb_size, transb=True):
    print(transb_size)
    return te.extern(
        (transb_size),
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

def batch_matmul_mlas(X, Y, transb=True, packb=False, K=-1, N=-1):
    XB, M, K = get_const_tuple(X.shape)
    if packb:
        YB = XB
    else:
        YB, N, K = get_const_tuple(Y.shape)

    assert XB == YB
    return te.extern(
        (XB, M, N),
        [X, Y],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.mlas.sgemm",
            M,
            N,
            K,
            packb,
            ins[0],
            ins[1],
            outs[0],
        ),
        name="C",
    )
