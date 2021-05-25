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
 * \file Use external mkl library call.
 */
#include <mlas.h>
#include <stdio.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.mlas.batch_matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  ICHECK(TypeMatch(A->dtype, kDLFloat, 32) || TypeMatch(A->dtype, kDLFloat, 64));
  size_t pack_b_size = MlasGemmPackBSize(100, 160);
  LOG(INFO) << pack_b_size;
  for (size_t N = 1; N < 100; N++) {
    const size_t AlignedN = (N + 16 - 1) & ~(16 - 1);
    std::cout << N << "--->" << AlignedN << std::endl;
  }
  LOG(INFO) << MlasGetPreferredBufferAlignment();
});

TVM_REGISTER_GLOBAL("tvm.contrib.mlas.gemm_packb_size")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      int N = args[0];
      int K = args[1];
      size_t packb_size = MlasGemmPackBSize(N, K);
      *ret = (int64_t)packb_size;
    });

TVM_REGISTER_GLOBAL("tvm.contrib.mlas.gemm_packb").set_body([](TVMArgs args, TVMRetValue* ret) {
  int N = args[0];
  int K = args[1];
  int ldb = args[2];
  bool transb = args[3];
  DLTensor* B = args[4];
  DLTensor* PackedB = args[5];
  std::cout << PackedB->data << std::endl;
  if (transb) {
    MlasGemmPackB(CblasTrans, N, K, (float*)B->data, ldb, (void*)PackedB->data);
  }
  else{
    MlasGemmPackB(CblasNoTrans, N, K, (float*)B->data, ldb, (void*)PackedB->data);
  }
  
});

TVM_REGISTER_GLOBAL("tvm.contrib.mlas.sgemm").set_body([](TVMArgs args, TVMRetValue* ret) {
  int M = args[0];
  int N = args[1];
  int K = args[2];
  bool packb = args[3];
  float alpha = 1.0;
  float beta = 0.0;
  DLTensor* A = args[4];
  DLTensor* B = args[5];
  DLTensor* C = args[6];
  std::cout << "M=" << M <<"N=" << N << "K=" << K << std::endl;
  if (packb) {
    MlasGemm(CblasNoTrans, static_cast<size_t>(M), static_cast<size_t>(N),
            static_cast<size_t>(K), alpha, (float*)A->data, static_cast<size_t>(K), B->data, beta,
            (float*)C->data, static_cast<size_t>(N));
  }
  else{
    MlasGemm(CblasNoTrans, CblasTrans, static_cast<size_t>(M), static_cast<size_t>(N),
            static_cast<size_t>(K), alpha, (float*)A->data, static_cast<size_t>(K), (float*)B->data, static_cast<size_t>(K), beta,
            (float*)C->data, static_cast<size_t>(N));
  }
});

}  // namespace contrib
}  // namespace tvm
