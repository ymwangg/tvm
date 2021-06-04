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
    if len(A.shape) == 3:
        batch_A, M_A, K_A = get_const_tuple(A.shape)
        if packb:
            # when B is packed, the batch_size must be 1
            batch_B, N_B, K_B = 1, in_n, in_k
        else:
            batch_B, N_B, K_B = get_const_tuple(B.shape)
        assert K_A == K_B
        assert (batch_A == batch_B) or (batch_B == 1)
        M, N, K = M_A, N_B, K_A
        return te.extern(
            (batch_A, M, N),
            [A, B],
            lambda ins, outs: tvm.tir.call_packed(
                "tvm.contrib.mlas.batch_sgemm",
                batch_A,
                batch_B,
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
    else:
        M_A, K_A = get_const_tuple(A.shape)
        if packb:
            N_B, K_B = in_n, in_k
        else:
            N_B, K_B = get_const_tuple(B.shape)
        assert K_A == K_B
        M, N, K = M_A, N_B, K_A
        return te.extern(
            (M, N),
            [A, B],
            lambda ins, outs: tvm.tir.call_packed(
                "tvm.contrib.mlas.batch_sgemm",
                1,
                1,
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
