from . import op as reg
from tvm.target import override_native_generic_func, Target, generic_func
from tvm.topi.nn import dense_alter_layout
from tvm import relay, topi
from .strategy import wrap_topi_schedule
import logging

# Mlas_matmul
# Mlas_matmul strategy
@override_native_generic_func("mlas_matmul_strategy")
def mlas_matmul_strategy(attrs, inputs, out_type, target):
    """mlas_matmul generic strategy"""
    return None


@mlas_matmul_strategy.register(["cpu", "arm_cpu"])
def mlas_matmul_strategy_cpu(attrs, inputs, out_type, target):
    """mlas_matmul strategy"""
    strategy = reg.OpStrategy()

    def wrap_compute_mlas_matmul(topi_compute):
        """wrap mlas_matmul topi compute"""

        def _compute_mlas_matmul(attrs, inputs, out_type):
            args = [inputs[0], inputs[1], attrs.packb, attrs.K, attrs.N]
            return [topi_compute(*args)]

        return _compute_mlas_matmul

    strategy.add_implementation(
        wrap_compute_mlas_matmul(topi.mlas_matmul),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="mlas_matmul",
        plevel=1,
    )
    return strategy


reg.register_strategy("mlas_matmul", mlas_matmul_strategy)
reg.register_pattern("mlas_matmul", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


# Mlas_matmul AlterOpLayout
@generic_func
def batch_matmul_alter_layout(attrs, inputs, tinfos, out_type):
    """Change batch_matmul layout."""
    # not to change by default
    return None


@batch_matmul_alter_layout.register(["cpu", "arm_cpu"])
def _alter_batch_matmul_layout(attrs, inputs, tinfos, out_type):
    target = Target.current(allow_none=False)
    if (
        "mlas" in target.libs
        and isinstance(inputs[1], relay.expr.Constant)
        and tinfos[0].dtype == "float32"
        and tinfos[1].dtype == "float32"
        and out_type.dtype == "float32"
    ):
        b_shape = inputs[1].data.shape
        assert len(b_shape) == 3
        batch, N, K = b_shape[0], b_shape[1], b_shape[2]
        assert batch == 1
        newb = relay.op.mlas_packb(inputs[1], K, N)
        output = relay.op.mlas_matmul(inputs[0], newb, True, K, N)
        logging.info("Applying mlas batch_matmul pack optimization for B.shape=", b_shape)
        return output
    return None


@reg.register_alter_op_layout("nn.batch_matmul")
def alter_op_layout_dense(attrs, inputs, tinfos, out_type):
    """Alternate the layout of batch_matmul"""
    return batch_matmul_alter_layout(attrs, inputs, tinfos, out_type)


# Dense
# Dense strategy
@override_native_generic_func("mlas_packb_strategy")
def mlas_packb_strategy(attrs, inputs, out_type, target):
    """mlas_packb generic strategy"""
    strategy = reg.OpStrategy()

    def wrap_mlas_packb(topi_compute):
        """Wrap mlas_packb topi compute"""

        def _compute_mlas_packb(attrs, inputs, _):
            return [topi_compute(inputs[0], attrs.K, attrs.N, attrs.size, attrs.transb)]

        return _compute_mlas_packb

    strategy.add_implementation(
        wrap_mlas_packb(topi.mlas_packb),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="mlas_packb",
    )
    return strategy


reg.register_strategy("mlas_packb", mlas_packb_strategy)

# Dense AlterOpLayout
# It overrides the original implementation in tvm.topi.x86.dense_alter_op
@dense_alter_layout.register(["cpu", "arm_cpu"], override=True)
def _alter_dense_layout(attrs, inputs, tinfos, out_type):
    target = Target.current(allow_none=False)
    if (
        "mlas" in target.libs
        and isinstance(inputs[1], relay.expr.Constant)
        and tinfos[0].dtype == "float32"
        and tinfos[1].dtype == "float32"
        and out_type.dtype == "float32"
    ):
        b_shape = inputs[1].data.shape
        assert len(b_shape) == 2
        N, K = b_shape[0], b_shape[1]
        newb = relay.op.mlas_packb(inputs[1], K, N)
        output = relay.op.mlas_matmul(inputs[0], newb, True, K, N)
        logging.info("Applying mlas dense pack optimization for B.shape=", b_shape)
        return output
    else:
        # default AlterOpLayout function copied from tvm.topi.x86.dense_alter_op
        from tvm import autotvm, te
        from tvm.topi.utils import get_const_tuple
        from tvm.topi.x86.dense import _default_dense_pack_config

        dispatch_ctx = autotvm.task.DispatchContext.current
        data_tensor, weight_tensor = tinfos
        out_dtype = out_type.dtype
        M, K = get_const_tuple(data_tensor.shape)
        N, _ = get_const_tuple(weight_tensor.shape)

        impl, outs = relay.backend.compile_engine.select_implementation(
            relay.op.get("nn.dense"), attrs, tinfos, out_type, target
        )
        workload = autotvm.task.get_workload(outs)
        if workload:
            cfg = dispatch_ctx.query(target, workload)
            topi_impl = workload[0]
            if topi_impl == "dense_pack.x86":
                if cfg.is_fallback:
                    _default_dense_pack_config(cfg, M, N, K)
                packw_bn = cfg["tile_x"].size[-1]
                weight_layout = "NK%dn" % packw_bn
                new_weight = te.placeholder(
                    (N // packw_bn, K, packw_bn),
                    dtype=weight_tensor.dtype,
                )
                # Relay dense doesn't have bias.
                new_workload = autotvm.task.args_to_workload(
                    [
                        data_tensor,
                        new_weight,
                        None,
                        out_dtype,
                    ],
                    topi_impl,
                )
                dispatch_ctx.update(target, new_workload, cfg)
                weight_transform = relay.layout_transform(inputs[1], "NK", weight_layout)
                return relay.nn.contrib_dense_pack(inputs[0], weight_transform, None, out_dtype)
    return None