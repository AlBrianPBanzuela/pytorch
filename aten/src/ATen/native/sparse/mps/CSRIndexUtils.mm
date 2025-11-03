#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/mps/CSRIndexUtils.h>

#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/SparseTensorUtils.h>

#include <ATen/TensorOperators.h>

#include <ATen/ops/_validate_compressed_sparse_indices.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cumsum_native.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/remainder.h>
#include <ATen/ops/scatter_native.h>
#include <ATen/ops/zeros_native.h>

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace at {
void _validate_compressed_sparse_indices(
    bool is_crow,
    const Tensor& compressed_idx,
    const Tensor& plain_idx,
    int64_t cdim,
    int64_t dim,
    int64_t nnz);
} // namespace at

namespace at::native::mps::csr {

namespace {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& metal_lib() {
  return at::native::mps::MetalShaderLibrary::getBundledLibrary();
}
#else
#include <ATen/native/mps/SparseTensorMath_metallib.h>
static auto& metal_lib() {
  return at::native::mps::MetalShaderLibrary::getBundledLibrary();
}
#endif

} // namespace

using namespace mps;

void build_batch_ptr_mps_out(
    const Tensor& batch_indices,
    int64_t batch_count,
    const Tensor& batch_ptr) {
  TORCH_CHECK(
      batch_indices.is_mps() && batch_ptr.is_mps(),
      "build_batch_ptr_mps_out: expected MPS tensors");
  TORCH_CHECK(
      batch_ptr.scalar_type() == at::kLong,
      "build_batch_ptr_mps_out: expected output dtype int64 but got ",
      batch_ptr.scalar_type());
  TORCH_CHECK(
      batch_ptr.numel() == batch_count + 1,
      "build_batch_ptr_mps_out: expected output shape [",
      batch_count + 1,
      "] but got ",
      batch_ptr.numel());

  // Builds an array of pointers for where each batch begins/ends in a packed
  // COO index tensor. Example:
  //   batch indices: [0, 0, 0, 1, 1, 2, 2, 2, 2]
  //                   └─────┘  └──┘  └─────────┘
  //                   batch0  batch1   batch2
  //   batch_ptr -> [0, 3, 5, 9]
  //                  │  │  │  └─ end of batch2 (total nnz)
  //                  │  │  └──── batch2 starts at index 5
  //                  │  └─────── batch1 starts at index 3
  //                  └────────── batch0 starts at index 0

  auto* stream = getCurrentMPSStream();
  const int64_t nnz = batch_indices.numel();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = metal_lib().getPipelineStateForFunc("build_batch_ptr_from_sorted_batches");
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];

      const uint32_t tew = pso.threadExecutionWidth;
      const uint32_t Q = static_cast<uint32_t>(batch_count + 1);
      const uint32_t tgW = std::min<uint32_t>(Q, tew);

      MTLSize grid = MTLSizeMake(Q, 1, 1);
      MTLSize tgs = MTLSizeMake(tgW, 1, 1);

      mtl_setArgs(
          enc,
          batch_indices,
          batch_ptr,
          std::array<uint32_t, 2>{static_cast<uint32_t>(nnz),
                                   static_cast<uint32_t>(batch_count)});
      [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
    }
  });
}

Tensor build_batch_ptr_mps(const Tensor& batch_indices, int64_t batch_count) {
  auto options = batch_indices.options().dtype(at::kLong);
  Tensor batch_ptr = at::zeros({batch_count + 1}, options);
  build_batch_ptr_mps_out(batch_indices, batch_count, batch_ptr);
  return batch_ptr;
}

void build_row_ptr_per_batch_mps_out(
    const Tensor& rows,
    const Tensor& batch_ptr,
    int64_t batch_count,
    int64_t rows_per_batch,
    const Tensor& row_ptr) {
  TORCH_CHECK(
      rows.is_mps() && batch_ptr.is_mps() && row_ptr.is_mps(),
      "build_row_ptr_per_batch_mps_out: expected MPS tensors");
  TORCH_CHECK(
      batch_ptr.scalar_type() == at::kLong,
      "build_row_ptr_per_batch_mps_out: expected batch_ptr dtype int64 but got ",
      batch_ptr.scalar_type());
  TORCH_CHECK(
      row_ptr.scalar_type() == at::kLong,
      "build_row_ptr_per_batch_mps_out: expected row_ptr dtype int64 but got ",
      row_ptr.scalar_type());
  TORCH_CHECK(
      row_ptr.numel() == batch_count * (rows_per_batch + 1),
      "build_row_ptr_per_batch_mps_out: expected output shape [",
      batch_count * (rows_per_batch + 1),
      "] but got ",
      row_ptr.numel());

  // Builds per-batch CSR-style row pointer arrays from sorted row indices.
  // Example (B = 2, rows_per_batch = 4):
  //   rows      = [0, 0, 1, 3,   0, 2, 2]
  //                └─ batch0 ─┘  └ batch1 ─┘
  //   batch_ptr = [0, 4, 7]
  //   row_ptr   -> [0, 2, 3, 3, 4,   0, 1, 1, 3, 3]
  //                  row_ptr[0]          row_ptr[1]

  auto* stream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto pso = metal_lib().getPipelineStateForFunc("build_row_ptr_from_sorted_rows_by_batch");
      auto enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];

      const uint32_t tew = pso.threadExecutionWidth;
      const uint32_t Qx = static_cast<uint32_t>(rows_per_batch + 1);
      const uint32_t Qy = static_cast<uint32_t>(batch_count);
      const uint32_t tgW = std::min<uint32_t>(Qx, tew);

      MTLSize grid = MTLSizeMake(Qx, Qy, 1);
      MTLSize tgs = MTLSizeMake(tgW, 1, 1);

      mtl_setArgs(
          enc,
          rows,
          batch_ptr,
          row_ptr,
          std::array<uint32_t, 2>{static_cast<uint32_t>(rows_per_batch),
                                   static_cast<uint32_t>(batch_count)});
      [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
    }
  });
}

Tensor build_row_ptr_per_batch_mps(
    const Tensor& rows,
    const Tensor& batch_ptr,
    int64_t batch_count,
    int64_t rows_per_batch) {
  auto options = rows.options().dtype(at::kLong);
  Tensor row_ptr = at::empty({batch_count * (rows_per_batch + 1)}, options);
  build_row_ptr_per_batch_mps_out(rows, batch_ptr, batch_count, rows_per_batch, row_ptr);
  return row_ptr;
}

void expand_csr_rows_to_coo_out(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    int64_t rows_per_batch,
    bool out_int32,
    bool transpose,
    const Tensor& coo_indices) {
  TORCH_CHECK(
      crow_indices.is_mps() && col_indices.is_mps() && coo_indices.is_mps(),
      "expand_csr_rows_to_coo: expected MPS tensors");
  TORCH_CHECK(
      crow_indices.scalar_type() == at::kLong,
      "expand_csr_rows_to_coo: crow_indices must be int64");
  TORCH_CHECK(
      col_indices.scalar_type() == (out_int32 ? at::kInt : at::kLong),
      "expand_csr_rows_to_coo: col_indices dtype mismatch");

  TORCH_CHECK(crow_indices.dim() >= 1, "expand_csr_rows_to_coo: expected batched crow_indices");

  if (col_indices.numel() == 0) {
    coo_indices.zero_();
    return;
  }

  const int64_t rows_plus_one = crow_indices.size(-1);
  TORCH_CHECK(
      rows_plus_one == rows_per_batch + 1,
      "expand_csr_rows_to_coo: crow_indices last dimension must equal rows_per_batch + 1");

  auto batch_shape = crow_indices.sizes().slice(0, crow_indices.dim() - 1);
  const int64_t batch_size = std::accumulate(
      batch_shape.begin(), batch_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());

  TORCH_CHECK(
      col_indices.numel() % batch_size == 0,
      "expand_csr_rows_to_coo: col_indices elements must be divisible by batch count");
  const int64_t nnz_per_batch = col_indices.numel() / batch_size;

  const int64_t batch_ndim = static_cast<int64_t>(batch_shape.size());
  const int64_t expected_rows = batch_ndim + 2;
  const int64_t total_nnz = col_indices.numel();

  TORCH_CHECK(
      coo_indices.dim() == 2 &&
      coo_indices.size(0) == expected_rows &&
      coo_indices.size(1) == total_nnz,
      "expand_csr_rows_to_coo: output must have shape [",
      expected_rows,
      ", ",
      total_nnz,
      "]");

  Tensor crow_flat = crow_indices.reshape({batch_size, rows_plus_one}).contiguous();
  Tensor starts = crow_flat.slice(/*dim=*/1, /*start=*/0, /*end=*/rows_plus_one - 1);

  auto options_long = crow_indices.options().dtype(at::kLong);
  Tensor indicator = at::zeros({batch_size, nnz_per_batch}, options_long);
  if (rows_per_batch > 0 && nnz_per_batch > 0) {
    indicator.scatter_(1, starts, 1);
  }

  Tensor rows_flat = (at::cumsum(indicator, 1) - 1).reshape({total_nnz});
  Tensor cols_flat = col_indices.reshape({total_nnz}).contiguous();

  Tensor linear_matrix = at::arange(batch_size, options_long).unsqueeze(1).expand({batch_size, nnz_per_batch});
  Tensor linear_flat = linear_matrix.reshape({total_nnz});

  std::vector<int64_t> strides(batch_ndim);
  int64_t stride_acc = 1;
  for (int64_t i = batch_ndim - 1; i >= 0; --i) {
    strides[i] = stride_acc;
    stride_acc *= batch_shape[i];
  }

  for (int64_t dim_idx = 0; dim_idx < batch_ndim; ++dim_idx) {
    int64_t size = batch_shape[dim_idx];
    if (size == 1) {
      coo_indices.select(0, dim_idx).zero_();
      continue;
    }
    int64_t stride = strides[dim_idx];
    Tensor coord = at::floor_divide(linear_flat, stride);
    coord = at::remainder(coord, size);
    coo_indices.select(0, dim_idx).copy_(coord.to(coo_indices.scalar_type()));
  }

  auto assign_row = [&](int64_t idx, const Tensor& src) {
    Tensor tmp = src.scalar_type() == coo_indices.scalar_type()
        ? src
        : src.to(coo_indices.scalar_type());
    coo_indices.select(0, idx).copy_(tmp);
  };

  if (transpose) {
    assign_row(batch_ndim, cols_flat);
    assign_row(batch_ndim + 1, rows_flat);
  } else {
    assign_row(batch_ndim, rows_flat);
    assign_row(batch_ndim + 1, cols_flat);
  }
}

Tensor expand_csr_rows_to_coo(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    int64_t rows_per_batch,
    bool out_int32,
    bool transpose) {
  auto batch_shape = crow_indices.sizes().slice(0, crow_indices.dim() - 1);
  const int64_t batch_dim = std::accumulate(
      batch_shape.begin(), batch_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  const int64_t total_nnz = col_indices.numel();
  const int64_t nnz_per_batch = batch_dim > 0 ? total_nnz / std::max<int64_t>(batch_dim, int64_t{1}) : 0;
  const int64_t batch_ndim = static_cast<int64_t>(batch_shape.size());
  const int64_t expected_rows = batch_ndim + 2;
  auto options = crow_indices.options().dtype(out_int32 ? at::kInt : at::kLong);
  Tensor coo_indices = at::empty({expected_rows, total_nnz}, options);
  if (total_nnz == 0) {
    coo_indices.zero_();
    return coo_indices;
  }
  expand_csr_rows_to_coo_out(
      crow_indices,
      col_indices,
      rows_per_batch,
      out_int32,
      transpose,
      coo_indices);
  return coo_indices;
}

} // namespace at::native::mps::csr

namespace at::native {

void _validate_compressed_sparse_indices_mps(
    const bool is_crow,
    const Tensor& cidx,
    const Tensor& idx,
    const int64_t cdim,
    const int64_t dim,
    const int64_t nnz) {
  auto cidx_cpu = cidx.cpu();
  auto idx_cpu = idx.cpu();
  at::_validate_compressed_sparse_indices(
      is_crow,
      cidx_cpu,
      idx_cpu,
      cdim,
      dim,
      nnz);
}

} // namespace at::native


