/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/matmul_op.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"


namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;


__device__ static int res;
__device__ static int total;

template <typename T>
__global__ void FusedMatMulAddReluPreKernel(
    CudaLaunchConfig config,
    const T* A,
    const T* B,
    const T* C,
    const T* norm_B,
    const T* d_B_L,
    const T* L,
    int n,
    int m,
    int k,
    int num_landmarks,
    T* output) {

  //printf("invoke!\n");
    // Hard coded num_lanmarks
    #define NUM_LANDMARKS 49
    T d_A_L[NUM_LANDMARKS];
  // Add C to output and apply relu
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    T norm_squared = 0.0;
    for (int i=0; i<NUM_LANDMARKS; i++)
      d_A_L[i] = 0.0;

    for (int y=0; y<k; y++) {
      int idx = x * k + y;
      norm_squared += (A[idx])*(A[idx]);
      for (int i=0; i<NUM_LANDMARKS; i++) {
        d_A_L[i] += (A[idx] - L[i*k + y])*(A[idx] - L[i*k + y]);
      }
    }

    for (int j=0; j<m; j++) {
      atomicAdd(&total, 1);
      for (int i=0; i<NUM_LANDMARKS; i++) {
        //float tmp = sqrt(d_A_L[i]) + sqrt(d_B_L[i*m + j]);
        T tmp = sqrt(d_A_L[i]) - sqrt(d_B_L[i*m + j]);
        T ub = (0.5 * (norm_squared + norm_B[j] - tmp * tmp));
        T aub = sqrt(norm_squared) * sqrt(norm_B[j]);
        T relu_threshold = -C[j];
        if (ub < relu_threshold || aub < relu_threshold) {
          atomicAdd(&res, 1);
          printf("%f, %f, %f, succ\n", aub, ub, relu_threshold);
          break;
        } else {
          //printf("%f, %f, %f, fail\n", aub, ub, relu_threshold);
        }
      }
    }
  }
}

template <typename T>
__global__ void FusedMatMulAddReluPostKernel(
    Cuda2DLaunchConfig config,
    const T* A,
    const T* B,
    const T* C,
    const T* norm_B,
    const T* d_B_L,
    const T* L,
    int n,
    int m,
    int k,
    int num_landmarks,
    T* output) {
  // Add C to output and apply relu
  CUDA_AXIS_KERNEL_LOOP(x, config.virtual_thread_count, x) {
    CUDA_AXIS_KERNEL_LOOP(y, config.virtual_thread_count, y) {
      int idx = x * config.virtual_thread_count.y + y;
      output[idx] = max(0.0, output[idx] + C[y]);
      if (x==0 && y==0) {
        // int count = 0;
        // for (int i=0; i<10000*120; i++)
        //   count += res[i];
        printf("saving:%d\n", res);
        printf("total:%d\n", total);
      }
    }
  }
}

namespace functor {

template <typename T>
void FusedMatMulAddReluFunctor<T>::pre(
    const GPUDevice& d,
    const T* A,
    const T* B,
    const T* C,
    const T* norm_B,
    const T* d_B_L,
    const T* L,
    int m,
    int n,
    int k,
    int num_landmarks,
    T* output) {
  VLOG(3) << "Tian Jin: try to launch kernel\n";
  CudaLaunchConfig config = GetCudaLaunchConfig(m, d, FusedMatMulAddReluPreKernel<T>, 0, 1);
  FusedMatMulAddReluPreKernel<<<config.block_count, config.thread_per_block,
    0, d.stream()>>>(config, A, B, C, norm_B, d_B_L, L, m, n, k, num_landmarks, output);
  VLOG(3) << "Kernel Launch Finished";
};

template <typename T>
void FusedMatMulAddReluFunctor<T>::post(
    const GPUDevice& d,
    const T* A,
    const T* B,
    const T* C,
    const T* norm_B,
    const T* d_B_L,
    const T* L,
    int m,
    int n,
    int k,
    int num_landmarks,
    T* output) {
  Cuda2DLaunchConfig config = GetCuda2DLaunchConfig(m, n, d);
  FusedMatMulAddReluPostKernel<<<config.block_count, config.thread_per_block,
    0, d.stream()>>>(config, A, B, C, norm_B, d_B_L, L, m, n, k, num_landmarks, output);
};

} // namespace functor

#define DEFINE(T)                                \
  template struct functor::FusedMatMulAddReluFunctor<T>;

DEFINE(float)
#undef DEFINE

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
