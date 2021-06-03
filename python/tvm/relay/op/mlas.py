from . import _make


def mlas_matmul(x, y, packb=False, K=-1, N=-1):
    r"""
    Computes batch matrix multiplication of `x` and `y` when `x` and `y` are data
    in batch.

    .. math::

        \mbox{batch_matmul}(x, y)[i, :, :] = \mbox{matmul}(x[i, :, :], y[i, :, :]^T)

    Parameters
    ----------
    x : tvm.relay.Expr
        The first input.

    y : tvm.relay.Expr
        The second input.

    out_dtype : str, optional
        Specifies the output data type for mixed precision batch matmul

    Returns
    -------
    result: tvm.relay.Expr
        The computed result.
    """
    return _make.mlas_matmul(x, y, packb, K, N)


def mlas_packb(B, K, N, transb=True):
    """Transform the layout of a tensor

    Parameters
    ----------

    Returns
    -------
    """
    import tvm

    get_packb_size = tvm._ffi.get_global_func("tvm.contrib.mlas.gemm_packb_size")
    packb_size = get_packb_size(N, K)
    arr_size = int(packb_size / 4)
    return _make.mlas_packb(B, K, N, arr_size, transb)