#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>

#include <curand_kernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_uniform_native.h>
#include <ATen/ops/empty.h>
#endif

namespace at::native {

namespace {

// Dtype-specific uniform generation. The primary template handles
// float/half/bfloat16 via curand_uniform4; the double specialization uses
// curand_uniform_double.
template <typename scalar_t>
__device__ void uniform_generate(
    scalar_t* output, int64_t base, int64_t elem, int64_t elem_end,
    curandStatePhilox4_32_10_t* state, double low, double high) {
  float flow = static_cast<float>(low);
  float frange = static_cast<float>(high - low);
  float4 u = curand_uniform4(state);
  float vals[4] = {
    flow + frange * u.x, flow + frange * u.y,
    flow + frange * u.z, flow + frange * u.w
  };
  #pragma unroll
  for (int j = 0; j < 4 && elem + j < elem_end; j++) {
    output[base + elem + j] = static_cast<scalar_t>(vals[j]);
  }
}
template <>
__device__ void uniform_generate<double>(
    double* output, int64_t base, int64_t elem, int64_t elem_end,
    curandStatePhilox4_32_10_t* state, double low, double high) {
  double range = high - low;
  double u0 = curand_uniform_double(state);
  output[base + elem] = low + range * u0;
  if (elem + 1 < elem_end) {
    double u1 = curand_uniform_double(state);
    output[base + elem + 1] = low + range * u1;
  }
}

template <typename scalar_t>
__global__ void philox_uniform_kernel(
    scalar_t* __restrict__ output,
    const uint64_t* __restrict__ keys,
    int64_t num_keys,
    int64_t event_numel,
    int64_t elems_per_thread,
    double low,
    double high) {
  constexpr size_t compute_size =
      sizeof(scalar_t) < sizeof(float) ? sizeof(float) : sizeof(scalar_t);
  constexpr int outputs_per_value = compute_size / sizeof(float);
  constexpr int elems_per_call = 4 / outputs_per_value;

  int64_t num_chunks = (event_numel + elems_per_thread - 1) / elems_per_thread;
  int64_t total_threads = num_keys * num_chunks;
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (; tid < total_threads; tid += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t key_idx = tid / num_chunks;
    int64_t chunk_idx = tid % num_chunks;
    int64_t elem_start = chunk_idx * elems_per_thread;
    int64_t elem_end = min(elem_start + elems_per_thread, event_numel);

    uint64_t seed = keys[key_idx * 2];
    uint64_t offset = keys[key_idx * 2 + 1];

    curandStatePhilox4_32_10_t state;
    curand_init(seed, /*subsequence=*/0, /*offset=*/offset, &state);

    int64_t base = key_idx * event_numel;
    skipahead(
        static_cast<unsigned long long>(elem_start) * outputs_per_value,
        &state);
    for (int64_t elem = elem_start; elem < elem_end; elem += elems_per_call) {
      uniform_generate<scalar_t>(output, base, elem, elem_end, &state, low, high);
    }
  }
}

} // anonymous namespace

Tensor _philox_uniform_cuda(const Tensor& self, const Tensor& key, double low, double high) {
  TORCH_CHECK(key.dim() >= 1 && key.size(-1) == 2,
      "_philox_uniform: key must have shape (*batch, 2), got shape ",
      key.sizes());
  TORCH_CHECK(key.scalar_type() == kUInt64,
      "_philox_uniform: key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(key.is_cuda(),
      "_philox_uniform: key must be a CUDA tensor");
  TORCH_CHECK(self.is_cuda(),
      "_philox_uniform: self must be a CUDA tensor");
  TORCH_CHECK(self.is_floating_point(),
      "_philox_uniform: self must be a floating point tensor, got ",
      self.scalar_type());
  TORCH_CHECK(self.device() == key.device(),
      "_philox_uniform: self and key must be on the same device, got ",
      self.device(), " and ", key.device());

  int64_t key_batch_ndim = key.dim() - 1;
  TORCH_CHECK(self.dim() >= key_batch_ndim,
      "_philox_uniform: self must have at least ", key_batch_ndim,
      " dimensions to match key batch dims, got ", self.dim());

  for (int64_t i = 0; i < key_batch_ndim; i++) {
    TORCH_CHECK(key.size(i) == 1 || key.size(i) == self.size(i),
        "_philox_uniform: key batch dim ", i, " has size ", key.size(i),
        " which is incompatible with self dim size ", self.size(i));
  }

  at::cuda::CUDAGuard device_guard(key.device());

  // Expand key batch dims to match self, then make contiguous.
  std::vector<int64_t> expanded_key_sizes;
  expanded_key_sizes.reserve(key_batch_ndim + 1);
  for (int64_t i = 0; i < key_batch_ndim; i++) {
    expanded_key_sizes.push_back(self.size(i));
  }
  expanded_key_sizes.push_back(2);
  auto key_expanded = key.expand(expanded_key_sizes).contiguous();

  int64_t num_keys = key_expanded.numel() / 2;
  int64_t event_numel = 1;
  for (int64_t i = key_batch_ndim; i < self.dim(); i++) {
    event_numel *= self.size(i);
  }

  Tensor output = at::empty(self.sizes(), self.options());

  if (num_keys == 0 || event_numel == 0) {
    return output;
  }

  constexpr int64_t elems_per_thread = 16;
  int64_t num_chunks = (event_numel + elems_per_thread - 1) / elems_per_thread;
  int64_t total_threads = num_keys * num_chunks;
  constexpr int block_size = 256;
  int num_blocks = std::min(
      static_cast<int>((total_threads + block_size - 1) / block_size),
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 4);

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "_philox_uniform_cuda", [&] {
    philox_uniform_kernel<scalar_t><<<num_blocks, block_size, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<scalar_t>(),
        key_expanded.data_ptr<uint64_t>(),
        num_keys, event_numel, elems_per_thread, low, high);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return output;
}

} // namespace at::native
