import tvm
from tvm import relay
m = 768
k = 768
n = 3072
A = relay.var('A',shape=[1,m,k])
B = relay.var('B',shape=[1,n,k])
B_packed = relay.mlas_packb(B, k, n)
C = relay.nn.mlas_matmul(A,B_packed,True,k,n)
f = relay.Function([A,B],C)
mod = tvm.IRModule({"main": f})

dev = tvm.cpu()
target = 'llvm'
des_exec = relay.vm.compile(mod, target)
des_vm = tvm.runtime.vm.VirtualMachine(des_exec, dev)
