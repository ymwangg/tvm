from tvm import relay
import numpy as np
import tvm

k, n = 3072, 768
A = relay.var("A", shape=[k, n])
B = relay.mlas_packb(k, n, A, False)
f = relay.Function([A], B)
mod = tvm.IRModule({"main": f})
dev = tvm.cpu()
target = "llvm"
executable = relay.vm.compile(mod, target)
des_vm = tvm.runtime.vm.VirtualMachine(executable, dev)

a = np.random.random([k, n]).astype("float32")
b = des_vm.run(a)

