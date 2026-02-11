//  Copyright © 2026 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSBulkLoad.h>

#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mutex>

#import <Foundation/Foundation.h>

#include <nlohmann/json.hpp>
#include <cerrno>

namespace at::mps {

namespace {

ssize_t safe_pread(int fd, void* buf, size_t count, off_t offset) {
  char* ptr = static_cast<char*>(buf);
  size_t total = 0;

  while (total < count) {
    ssize_t result = pread(fd, ptr + total, count - total, offset + total);
    if (result < 0) {
      if (errno == EINTR)
        continue;
      return -1;
    }
    if (result == 0)
      break;
    total += result;
  }
  return total;
}

ssize_t safe_read(int fd, void* buf, size_t count) {
  char* ptr = static_cast<char*>(buf);
  size_t remaining = count;
  ssize_t total = 0;

  while (remaining > 0) {
    ssize_t result = ::read(fd, ptr, remaining);

    if (result == -1) {
      if (errno == EINTR)
        continue;
      return -1;
    }

    if (result == 0)
      break;

    ptr += result;
    remaining -= result;
    total += result;
  }

  return total;
}

c10::ScalarType safetensors_dtype_to_scalar_type(const std::string& dtype_str) {
  static const std::unordered_map<std::string, c10::ScalarType> dtype_map = {
      {"F64", c10::ScalarType::Double},
      {"F32", c10::ScalarType::Float},
      {"F16", c10::ScalarType::Half},
      {"BF16", c10::ScalarType::BFloat16},
      {"I64", c10::ScalarType::Long},
      {"I32", c10::ScalarType::Int},
      {"I16", c10::ScalarType::Short},
      {"I8", c10::ScalarType::Char},
      {"U8", c10::ScalarType::Byte},
      {"BOOL", c10::ScalarType::Bool},
      {"F8_E5M2", c10::ScalarType::Float8_e5m2},
      {"F8_E4M3", c10::ScalarType::Float8_e4m3fn},
  };

  auto it = dtype_map.find(dtype_str);
  TORCH_CHECK(it != dtype_map.end(), "Unsupported safetensors dtype: ", dtype_str);
  return it->second;
}

} // anonymous namespace

MPSBulkLoader::MPSBulkLoader(const std::string& filename) : filename_(filename) {
  fd_ = open(filename_.c_str(), O_RDONLY);
  TORCH_CHECK(fd_ >= 0, "MPSBulkLoader: unable to open file: ", filename_, ", error: ", strerror(errno));
}

MPSBulkLoader::~MPSBulkLoader() {
  if (fd_ >= 0) {
    close(fd_);
  }
}

void MPSBulkLoader::parallelRead(const std::vector<TensorLoadInfo>& tensor_infos, std::vector<void*>& buffers) {
  const size_t count = tensor_infos.size();
  if (count == 0)
    return;

  for (const auto& info : tensor_infos) {
    TORCH_CHECK(info.file_offset < SIZE_MAX - info.size_bytes, "File offset overflow for tensor: ", info.name);
  }

  dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

  auto had_error = std::make_shared<std::atomic<bool>>(false);
  auto error_message = std::make_shared<std::string>();
  auto error_mutex = std::make_shared<std::mutex>();

  dispatch_apply(count, queue, ^(size_t i) {
    const TensorLoadInfo& info = tensor_infos[i];
    ssize_t bytes_read = safe_pread(fd_, buffers[i], info.size_bytes, info.file_offset);
    int saved_errno = errno;

    if (bytes_read != static_cast<ssize_t>(info.size_bytes)) {
      if (!had_error->load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(*error_mutex);
        bool expected = false;
        if (had_error->compare_exchange_strong(expected, true, std::memory_order_release)) {
          *error_message = "Failed to read tensor '" + info.name + "': expected " + std::to_string(info.size_bytes) +
              " bytes, got " + std::to_string(bytes_read);
          if (bytes_read < 0) {
            *error_message += ", error: " + std::string(strerror(saved_errno));
          }
        }
      }
    }
  });

  TORCH_CHECK(!had_error->load(std::memory_order_acquire), "MPSBulkLoader: ", *error_message);
}

std::unordered_map<std::string, at::Tensor> MPSBulkLoader::loadTensors(
    const std::vector<TensorLoadInfo>& all_tensor_infos) {
  if (all_tensor_infos.empty()) {
    return {};
  }

  std::unordered_map<std::string, at::Tensor> result;
  result.reserve(all_tensor_infos.size());

  std::vector<void*> buffers;
  std::vector<TensorLoadInfo> tensor_infos;

  // Need to initialize all tensors (even ones with zero elements)
  for (const auto& info : all_tensor_infos) {
    at::Tensor tensor = at::empty(info.shape, at::TensorOptions().dtype(info.dtype).device(c10::DeviceType::MPS, 0));
    result.emplace(info.name, std::move(tensor));
  }

  // Only load the ones that have more than zero elements
  for (const auto& info : all_tensor_infos) {
    if (info.size_bytes > 0) {
      auto& tensor = result.at(info.name);

      auto* allocator = at::mps::getIMPSAllocator();
      void* ptr = allocator->getWritableSharedBufferPtr(tensor.data_ptr());
      TORCH_CHECK(
          ptr != nullptr, "MPSBulkLoader: Failed to get writable shared buffer pointer for tensor: ", info.name);

      buffers.emplace_back(ptr);
      tensor_infos.emplace_back(info);
    }
  }

  if (!tensor_infos.empty()) {
    parallelRead(tensor_infos, buffers);
  }

  return result;
}

std::vector<TensorLoadInfo> MPSBulkLoader::parseHeader(size_t& header_size_out) {
  uint64_t header_len = 0;
  ssize_t bytes_read = safe_read(fd_, &header_len, sizeof(header_len));
  TORCH_CHECK(bytes_read == sizeof(header_len), "Failed to read safetensors header size from: ", filename_);
  TORCH_CHECK(header_len > 0 && header_len < 100 * 1024 * 1024, "Invalid safetensors header size: ", header_len);

  std::vector<char> header_buf(header_len + 1);
  bytes_read = safe_read(fd_, header_buf.data(), header_len);
  TORCH_CHECK(bytes_read == static_cast<ssize_t>(header_len), "Failed to read safetensors header content");
  header_buf[header_len] = '\0';

  nlohmann::json header;
  try {
    header = nlohmann::json::parse(header_buf.data());
  } catch (const std::exception& e) {
    TORCH_CHECK(false, "Failed to parse safetensors header from ", filename_, ": ", e.what());
  }

  const size_t data_start = sizeof(uint64_t) + header_len;
  header_size_out = data_start;

  std::vector<TensorLoadInfo> tensor_infos;
  tensor_infos.reserve(header.size());

  for (const auto& [key, value] : header.items()) {
    if (key == "__metadata__") {
      continue;
    }

    TensorLoadInfo info;
    info.name = key;

    auto dtype_it = value.find("dtype");
    TORCH_CHECK(dtype_it != value.end(), "Missing dtype for tensor: ", key);
    info.dtype = safetensors_dtype_to_scalar_type(dtype_it->get<std::string>());

    auto shape_it = value.find("shape");
    TORCH_CHECK(shape_it != value.end(), "Missing shape for tensor: ", key);
    info.shape.reserve(shape_it->size());
    for (const auto& dim : *shape_it) {
      int64_t dimension = dim.get<int64_t>();
      TORCH_CHECK(dimension >= 0, "Invalid negative dimension in tensor: ", key);
      info.shape.push_back(dimension);
    }

    auto offsets_it = value.find("data_offsets");
    TORCH_CHECK(offsets_it != value.end(), "Missing data_offsets for tensor: ", key);
    TORCH_CHECK(offsets_it->size() == 2, "Invalid data_offsets format for tensor: ", key);

    size_t start_offset = (*offsets_it)[0].get<size_t>();
    size_t end_offset = (*offsets_it)[1].get<size_t>();
    TORCH_CHECK(end_offset >= start_offset, "Invalid offset range for tensor: ", key);

    info.file_offset = data_start + start_offset;
    info.size_bytes = end_offset - start_offset;

    TORCH_CHECK(start_offset <= SIZE_MAX - data_start, "Start offset overflow for tensor: ", key);
    TORCH_CHECK(end_offset <= SIZE_MAX - data_start, "End offset overflow for tensor: ", key);

    tensor_infos.emplace_back(std::move(info));
  }

  return tensor_infos;
}

std::unordered_map<std::string, at::Tensor> mps_load_safetensors(const std::string& filename) {
  MPSBulkLoader loader(filename);
  size_t header_size;
  auto tensor_infos = loader.parseHeader(header_size);
  return loader.loadTensors(tensor_infos);
}

} // namespace at::mps
