#include "elementwise.cuh"

namespace tiny_llm {
namespace kernels {

namespace {
__global__ void add_inplace_kernel(half *data, const half *add,
                                   int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float a = __half2float(data[idx]);
    float b = __half2float(add[idx]);
    data[idx] = __float2half(a + b);
  }
}

__global__ void silu_mul_inplace_kernel(half *gate, const half *up,
                                        int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu = g / (1.0f + expf(-g));
    gate[idx] = __float2half(silu * u);
  }
}
} // namespace

void add_inplace(half *data, const half *add, int num_elements,
                 cudaStream_t stream) {
  if (num_elements <= 0)
    return;
  int block_size = 256;
  int grid_size = (num_elements + block_size - 1) / block_size;
  add_inplace_kernel<<<grid_size, block_size, 0, stream>>>(data, add,
                                                           num_elements);
}

void silu_mul_inplace(half *gate, const half *up, int num_elements,
                      cudaStream_t stream) {
  if (num_elements <= 0)
    return;
  int block_size = 256;
  int grid_size = (num_elements + block_size - 1) / block_size;
  silu_mul_inplace_kernel<<<grid_size, block_size, 0, stream>>>(gate, up,
                                                                num_elements);
}

} // namespace kernels
} // namespace tiny_llm
