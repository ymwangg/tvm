import tvm
import numpy as np
from tvm import topi


def extract_shared_efficiency(path):
    with open(path) as fh:
        txt = fh.readlines()
    efficiency = -1.0
    for line in txt:
        line = line.strip()
        if line[-1] == "%":
            eff = float(line.split()[-1][:-1])
            if eff != 0:
                efficiency = eff
    if efficiency <= 0:
        raise RuntimeError("error parsing shared memory")
    return efficiency


def extract_bank_conflict(path):
    with open(path) as fh:
        txt = fh.readlines()
    res = {
        "shared_ld_bank_conflict": 0,
        "shared_st_bank_conflict": 0,
        "shared_ld_transactions": 0,
        "shared_st_transactions": 0,
        "shared_load": 0,
        "shared_store": 0,
    }
    for line in txt:
        for k in res.keys():
            if k in line:
                splitted = line.split()
                assert splitted[1] == k
                res[k] += int(splitted[-2])
    return res


class BankConflictCounter:
    """Helper class to simulate the bank conflict effect."""

    def __init__(self, ir):
        self.module = ir["main"].body
        # helper vars
        self.thread_block = {}
        self.for_level = 0
        self.for_product = 1
        self.expr_map = set()
        self.for_extent_list = []
        # memory ops
        self.local_st = 0
        self.local_ld = 0
        self.shared_st = 0
        self.shared_ld = 0
        self.global_st = 0
        self.global_ld = 0
        self.num_threads = 1
        # bank conflict
        self.shared_st_bank_conflict = 0
        self.shared_ld_bank_conflict = 0
        self._compute()

    def _compute(self):
        def _pre_visit(node):
            if isinstance(node, tvm.tir.stmt.For):
                self.for_level += 1
                self.for_extent_list.append(node.extent.value)
                self.for_product *= node.extent.value
            if isinstance(node, tvm.tir.expr.Load):
                expr = ("load", str(node.buffer_var.name), str(node.index))
                if expr not in self.expr_map:
                    if "shared" in node.buffer_var.name:
                        self.shared_ld += self.for_product
                        self.shared_ld_bank_conflict += (
                            self.for_product * self._compute_bank_conflict(node.index)
                        )
                    elif "local" in node.buffer_var.name:
                        self.local_ld += self.for_product
                    else:
                        self.global_ld += self.for_product
                    self.expr_map.add(expr)
            if isinstance(node, tvm.tir.stmt.Store):
                expr = ("store", str(node.buffer_var.name), str(node.index))
                if expr not in self.expr_map:
                    if "shared" in node.buffer_var.name:
                        self.shared_st += self.for_product
                        self.shared_st_bank_conflict += (
                            self.for_product * self._compute_bank_conflict(node.index)
                        )
                    elif "local" in node.buffer_var.name:
                        self.local_st += self.for_product
                    else:
                        self.global_st += self.for_product
                    self.expr_map.add(expr)
            if isinstance(node, tvm.tir.stmt.AttrStmt) and node.attr_key == "thread_extent":
                self.thread_block[node.node.var.name] = node.value.value

        def _post_visit(node):
            if isinstance(node, tvm.tir.stmt.For):
                self.for_level -= 1
                last = self.for_extent_list.pop()
                self.for_product /= last

        tvm.tir.stmt_functor.ir_transform(self.module.body, _pre_visit, _post_visit)

    def _compute_bank_conflict(self, node):
        x_extent = self.thread_block["threadIdx.x"]
        y_extent = self.thread_block["threadIdx.y"]
        z_extent = self.thread_block["threadIdx.z"]
        feed_dict = {}
        thread_dict = {}

        def _visit(node):
            if isinstance(node, tvm.tir.expr.Var):
                feed_dict[node] = 0
                if node.name == "threadIdx.x":
                    thread_dict["threadIdx.x"] = node
                elif node.name == "threadIdx.y":
                    thread_dict["threadIdx.y"] = node
                elif node.name == "threadIdx.z":
                    thread_dict["threadIdx.z"] = node

        tvm.tir.stmt_functor.post_order_visit(node, _visit)
        bank = np.zeros(32)
        # only consider unique indices due to intra-warp broadcasting
        unique = set()
        # only calculate the indices of the first warp
        for i in range(min(32, x_extent * y_extent * z_extent)):
            x = i % x_extent
            y = (i // x_extent) % y_extent
            z = (i // (x_extent * y_extent)) % z_extent
            if "threadIdx.x" in thread_dict:
                feed_dict[thread_dict["threadIdx.x"]] = x
            if "threadIdx.y" in thread_dict:
                feed_dict[thread_dict["threadIdx.y"]] = y
            if "threadIdx.z" in thread_dict:
                feed_dict[thread_dict["threadIdx.z"]] = z
            idx = tvm.topi.utils.simplify(tvm.tir.stmt_functor.substitute(node, feed_dict))
            if idx.value not in unique:
                bank[idx.value % 32] += 1
                unique.add(idx.value)
        return np.max(bank)

    def _compute_bank_conflict_factor(self):
        """Return the bank conflict factor"""
        return (self.shared_ld_bank_conflict + self.shared_st_bank_conflict) / (
            self.shared_ld + self.shared_st
        )
