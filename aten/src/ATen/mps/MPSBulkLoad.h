//  Copyright © 2026 Apple Inc.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace at::mps {

struct TensorLoadInfo {
  std::string name;
  c10::ScalarType dtype;
  std::vector<int64_t> shape;
  size_t file_offset;
  size_t size_bytes;
};

/**
 * MPSBulkLoader - Optimized bulk loading of tensors to MPS device
 *
 * This class provides high-performance loading of multiple tensors from
 * safetensors directly to MPS device using:
 *
 * 1. Parallel pread() via Grand Central Dispatch (GCD) for concurrent I/O
 * 2. Direct Metal buffer creation to minimize copies
 */
class TORCH_API MPSBulkLoader {
 public:
  explicit MPSBulkLoader(const std::string& filename);
  ~MPSBulkLoader();

  MPSBulkLoader(const MPSBulkLoader&) = delete;
  MPSBulkLoader& operator=(const MPSBulkLoader&) = delete;
  MPSBulkLoader(MPSBulkLoader&&) = delete;
  MPSBulkLoader& operator=(MPSBulkLoader&&) = delete;

  std::vector<TensorLoadInfo> parseHeader(size_t& header_size_out);

  std::unordered_map<std::string, at::Tensor> loadTensors(
      const std::vector<TensorLoadInfo>& tensor_infos);

 private:
  std::string filename_;
  int fd_ = -1;

  void parallelRead(
      const std::vector<TensorLoadInfo>& tensor_infos,
      std::vector<void*>& buffers);
};

TORCH_API std::unordered_map<std::string, at::Tensor> mps_load_safetensors(
    const std::string& filename);

} // namespace at::mps
