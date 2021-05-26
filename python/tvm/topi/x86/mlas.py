from numpy import MAY_SHARE_BOUNDS
from tvm import te
import tvm
from tvm.topi.utils import get_const_float, get_const_tuple


def mlas_packb(B, K, N, transb_size, transb=True):
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

def mlas_matmul(A, B, packb=False, in_k=0, in_n=0):

    batch_A, M_A, K_A = get_const_tuple(A.shape)
    if packb:
        batch_B, N_B, K_B = batch_A, in_n, in_k
    else:
        batch_B, N_B, K_B = get_const_tuple(B.shape)
    assert batch_A == batch_B
    assert K_A == K_B
    batch, M, N, K = batch_A, M_A, N_B, K_A
    return te.extern(
        (batch, M, N),
        [A, B],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.mlas.batch_sgemm",
            batch,
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
