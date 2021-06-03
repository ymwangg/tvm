from . import op as reg
from tvm.target import override_native_generic_func
from tvm.topi.nn import dense_alter_layout
from tvm import relay, topi
from .strategy import wrap_topi_schedule

# Mlas_matmul
# Mlas_matmul strategy
@override_native_generic_func("mlas_matmul_strategy")
def mlas_matmul_strategy(attrs, inputs, out_type, target):
    """mlas_matmul generic strategy"""
    strategy = reg.OpStrategy()
    return strategy

@mlas_matmul_strategy.register("cpu")
def mlas_matmul_strategy_cpu(attrs, inputs, out_type, target):
    """mlas_matmul x86 strategy"""
    strategy = reg.OpStrategy()

    def wrap_compute_mlas_matmul(topi_compute):
        """wrap mlas_matmul topi compute"""

        def _compute_mlas_matmul(attrs, inputs, out_type):
            args = [inputs[0], inputs[1], attrs.packb, attrs.K, attrs.N]
            return [topi_compute(*args)]

        return _compute_mlas_matmul
    strategy.add_implementation(
        wrap_compute_mlas_matmul(topi.x86.mlas_matmul),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="mlas_matmul.x86",
        plevel=15,
    )
    return strategy

reg.register_strategy("mlas_matmul", mlas_matmul_strategy)
reg.register_pattern("mlas_matmul", reg.OpPattern.OUT_ELEMWISE_FUSABLE)

# Mlas_matmul AlterOpLayout
@reg.register_alter_op_layout("nn.batch_matmul")
def alter_op_layout_dense(attrs, inputs, tinfos, out_type):
    """Alternate the layout of batch_matmul"""
    if isinstance(inputs[1], relay.expr.Constant):
        b_shape = inputs[1].data.shape
        assert len(b_shape) == 3
        batch, N, K = b_shape[0], b_shape[1], b_shape[2]
        assert batch == 1
        print("rewriting batch-matmul b_shape=", b_shape)
        newb = relay.op.mlas_packb(inputs[1], K, N)
        output = relay.op.mlas_matmul(inputs[0], newb, True, K, N)
        return output
    return None


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
        wrap_mlas_packb(topi.x86.mlas_packb),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="mlas_packb.x86",
    )
    return strategy

reg.register_strategy("mlas_packb", mlas_packb_strategy)

# Dense AlterOpLayout
@dense_alter_layout.register(["cpu"], override=True)
def _alter_dense_layout(attrs, inputs, tinfos, out_type):
    from tvm import relay

    if isinstance(inputs[1], relay.expr.Constant):
        b_shape = inputs[1].data.shape
        assert len(b_shape) == 2
        N, K = b_shape[0], b_shape[1]
        print("rewriting dense b_shape=", b_shape)
        newb = relay.op.mlas_packb(inputs[1], K, N)
        output = relay.op.mlas_matmul(inputs[0], newb, True, K, N)
        return output
    return None