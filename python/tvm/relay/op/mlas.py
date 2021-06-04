from . import _make
from tvm import _ffi


def mlas_matmul(A, B, packb=False, K=-1, N=-1):
    r"""
    Computes batch matrix multiplication of `A` and `B` when `A` and `B` are data
    in batch.

    .. math::

        C[i, :, :] = \mbox{matmul}(A[i, :, :], B[i, :, :]^T)

    Parameters
    ----------
    A : tvm.relay.Expr
        The first input.

    B : tvm.relay.Expr
        The second input.

    packb : bool
        Specify whether the B is pre-packed

    K : int
        The number of colums of A

    N : int
        The number of colums of output C

    Returns
    -------
    result: tvm.relay.Expr
        The computed result.
    """
    return _make.mlas_matmul(A, B, packb, K, N)


def mlas_packb(B, K, N, transb=True):
    """Pre-pack B matrix if it is constant for mlas_matmul, C = A * B^T

    Parameters
    ----------
    B : tvm.relay.Expr
        The second input of mlas_matmul

    packb : bool
        Specify whether the B is pre-packed

    K : int
        The number of colums of A

    N : int
        The number of colums of output C

    transb : bool
        Whether the B matrix is transposed
    Returns
    -------
    result: tvm.relay.Expr
        The pre-packed B matrix
    """
    get_packb_size = _ffi.get_global_func("tvm.contrib.mlas.gemm_packb_size")
    packb_size = get_packb_size(N, K)
    # only support float32
    arr_size = int(packb_size / 4)
    return _make.mlas_packb(B, K, N, arr_size, transb)