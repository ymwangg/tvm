import tvm
from tvm import relay
A = relay.var('A',shape=[1,100,100])
B = relay.var('B',shape=[1,100,100])
C = relay.nn.mlas_matmul(A,B)
f = relay.Function([A,B],C)
mod = relay.build(f, 'llvm')
