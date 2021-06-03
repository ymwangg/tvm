/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file mlas_op.cc
 * \brief Implementation of operators from MLAS library
 */


#include <tvm/relay/attrs/mlas_op.h>
#include <string>
#include <vector>

#include "../../transforms/infer_layout_utils.h"
#include "../make_op.h"
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

// relay.mlas_matmul
TVM_REGISTER_NODE_TYPE(MlasMatmulAttrs);

bool MlasMatmulRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* x = types[0].as<TensorTypeNode>();
  const auto* y = types[1].as<TensorTypeNode>();
  if (x == nullptr || y == nullptr) return false;
  const auto* param = attrs.as<MlasMatmulAttrs>();

  bool is_dyn = false;
  Array<tvm::PrimExpr> oshape;
  if (!param->packb) {
    ICHECK_EQ(x->shape.size(), y->shape.size());
  }

  if (!param->packb) {
    if (x->shape.size() == 3) {
      for (size_t i = 0; i < 3; ++i) {
        if (x->shape[i].as<tir::AnyNode>() != nullptr ||
            y->shape[i].as<tir::AnyNode>() != nullptr) {
          is_dyn = true;
          oshape.push_back(Any());
        } else {
          if (i == 0) {
            oshape.push_back(max(x->shape[i], y->shape[i]));
          } else {
            oshape.push_back(x->shape[i]);
          }
        }
      }
      if (!is_dyn) {
        ICHECK(reporter->AssertEQ(x->shape[0], y->shape[0]) || reporter->AssertEQ(x->shape[0], 1) ||
               reporter->AssertEQ(y->shape[0], 1))
            << "BatchDot: batch dimensions don't match, "
            << " x shape=" << x->shape << ", y shape=" << y->shape;
        ICHECK(reporter->AssertEQ(x->shape[2], y->shape[2]))
            << "BatchDot: shapes of x and y is inconsistent, "
            << " x shape=" << x->shape << ", y shape=" << y->shape;

        oshape.Set(2, y->shape[1]);
      }
    } else {
      for (size_t i = 0; i < 2; ++i) {
        if (x->shape[i].as<tir::AnyNode>() != nullptr ||
            y->shape[i].as<tir::AnyNode>() != nullptr) {
          is_dyn = true;
          oshape.push_back(Any());
        } else {
          oshape.push_back(x->shape[i]);
        }
      }
      if (!is_dyn) {
        oshape.Set(1, y->shape[0]);
      }
    }

  } else {
    if (x->shape.size() == 3) {
      oshape.push_back(x->shape[0]);
      oshape.push_back(x->shape[1]);
      reporter->AssertEQ(x->shape[2], param->K);
      oshape.push_back(param->N);
    } else {
      oshape.push_back(x->shape[0]);
      oshape.push_back(param->N);
      reporter->AssertEQ(x->shape[1], param->K);
    }
  }
  // if (!is_dyn) {
  //   ICHECK(reporter->AssertEQ(x->shape[0], y->shape[0]) || reporter->AssertEQ(x->shape[0], 1) ||
  //          reporter->AssertEQ(y->shape[0], 1))
  //       << "BatchDot: batch dimensions don't match, "
  //       << " x shape=" << x->shape << ", y shape=" << y->shape;
  //   ICHECK(reporter->AssertEQ(x->shape[2], y->shape[2]))
  //       << "BatchDot: shapes of x and y is inconsistent, "
  //       << " x shape=" << x->shape << ", y shape=" << y->shape;

  //   oshape.Set(2, y->shape[1]);
  // }

  // assign output type
  reporter->Assign(types[2], TensorType(oshape, x->dtype));
  return true;
}

Expr MakeMlasMatmul(Expr x, Expr y, bool packb, int K, int N) {
  auto attrs = make_object<MlasMatmulAttrs>();
  attrs->packb = packb;
  attrs->K = K;
  attrs->N = N;
  LOG(INFO) << "packb=" << packb << " K=" << K << " N=" << N;
  static const Op& op = Op::Get("mlas_matmul");
  return Call(op, {x, y}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.mlas_matmul").set_body_typed(MakeMlasMatmul);

RELAY_REGISTER_OP("mlas_matmul")
    .describe(R"code(Computes matrix multiplication using mlas library

.. math::

  batch\_matmul(x, y)[i, :, :] = matmul(x[i, :, :], y[i, :, :]^T)

- **x**: `(b, m, k)`
- **y**: `(b, n, k)`
- **out**: `(b, m, n)`.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("x", "3D Tensor", "First input.")
    .add_argument("y", "3D Tensor", "Second input.")
    .set_support_level(10)
    .add_type_rel("MlasMatmul", MlasMatmulRel);


// relay.mlas_packb
TVM_REGISTER_NODE_TYPE(MlasPackbAttrs);

bool MlasPackbRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  const auto* B = types[0].as<TensorTypeNode>();
  if (B == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "MlasPackbRel: expect input data type to be TensorType but get " << types[0];
    return false;
  }
  const MlasPackbAttrs* params = attrs.as<MlasPackbAttrs>();
  reporter->Assign(types[1], TensorType({params->size}, B->dtype));
  return true;
}

Expr MakeMlasPackb(Expr B, int K, int N, int size, bool transb) {
  auto attrs = make_object<MlasPackbAttrs>();
  attrs->K = K;
  attrs->N = N;
  attrs->size = size;
  attrs->transb = transb;
  static const Op& op = Op::Get("mlas_packb");
  return Call(op, {B}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.mlas_packb").set_body_typed(MakeMlasPackb);

RELAY_REGISTER_OP("mlas_packb")
    .describe(R"code(Pack the B matrix
)code" TVM_ADD_FILELINE)
    .set_attrs_type<MlasPackbAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_type_rel("mlas_packb", MlasPackbRel)
    .set_support_level(5)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace relay
}  // namespace tvm
