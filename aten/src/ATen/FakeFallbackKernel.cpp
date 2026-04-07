#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/empty_strided.h>
#include <c10/core/impl/FakeTensorModeTLS.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/irange.h>
#include <torch/library.h>

namespace {

static std::optional<c10::Device> get_common_device(
    torch::jit::Stack* stack,
    size_t num_arguments) {
  std::optional<c10::Device> common_device;
  bool is_cpu_zero_dim = false;

  auto merge = [&](const at::Tensor& t) {
    if (!t.defined() || !t.is_fake())
      return;
    bool t_is_cpu_zero_dim = t.device().is_cpu() && t.dim() == 0;
    if (!common_device.has_value()) {
      common_device = t.device();
      is_cpu_zero_dim = t_is_cpu_zero_dim;
      return;
    }
    if (t.device() == *common_device) {
      if (is_cpu_zero_dim)
        is_cpu_zero_dim = t_is_cpu_zero_dim;
      return;
    }
    if (t_is_cpu_zero_dim)
      return;
    TORCH_CHECK(
        is_cpu_zero_dim,
        "Unhandled FakeTensor device propagation: ",
        *common_device,
        " vs ",
        t.device());
    common_device = t.device();
    is_cpu_zero_dim = false;
  };

  auto arguments = torch::jit::last(*stack, num_arguments);
  for (size_t idx = 0; idx < num_arguments; ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      merge(ivalue.toTensor());
    } else if (ivalue.isTensorList()) {
      for (const auto& elem : ivalue.toTensorList())
        merge(elem);
    } else if (ivalue.isOptionalTensorList()) {
      for (const auto& elem : ivalue.toOptionalTensorList()) {
        std::optional<at::Tensor> ot = elem;
        if (ot.has_value())
          merge(*ot);
      }
    }
  }
  return common_device;
}

static bool is_device_type_arg(const c10::Argument& arg) {
  const auto& type = arg.type();
  if (type->kind() == c10::TypeKind::DeviceObjType)
    return true;
  if (type->kind() == c10::TypeKind::OptionalType) {
    auto elem = type->castRaw<c10::OptionalType>()->getElementType();
    return elem->kind() == c10::TypeKind::DeviceObjType;
  }
  return false;
}

static std::optional<c10::Device> rewrite_device_args_to_meta(
    torch::jit::Stack* stack,
    size_t arguments_begin,
    size_t num_arguments,
    const c10::FunctionSchema& schema) {
  std::optional<c10::Device> original_device;
  auto arguments = torch::jit::last(*stack, num_arguments);
  for (size_t idx = 0; idx < num_arguments; ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isDevice()) {
      auto dev = ivalue.toDevice();
      TORCH_CHECK(
          dev.type() != c10::DeviceType::Meta,
          "FakeTensor does not support meta device inputs");
      if (!original_device.has_value())
        original_device = dev;
      (*stack)[arguments_begin + idx] =
          c10::IValue(c10::Device(c10::DeviceType::Meta));
    } else if (ivalue.isNone() && is_device_type_arg(schema.arguments()[idx])) {
      if (!original_device.has_value()) {
        original_device = c10::Device(c10::DeviceType::CPU);
      }
      (*stack)[arguments_begin + idx] =
          c10::IValue(c10::Device(c10::DeviceType::Meta));
    }
  }
  return original_device;
}

static void transmute_to_fake(
    const at::Tensor& t,
    c10::Device fake_device,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  t.unsafeGetTensorImpl()->set_fake_device(fake_device);
  if (mode) {
    t.unsafeGetTensorImpl()->set_fake_tensor_mode(mode);
  }
}

static bool needs_transmute(const at::Tensor& t) {
  if (!t.defined())
    return false;
  if (!t.is_fake())
    return true;
  return t.device().is_meta();
}

// C++ equivalent of Python's validate_and_convert_non_fake_tensors.
// Converts real tensor inputs to C++ fake tensors by creating a meta tensor
// with matching shape/strides/dtype and stamping it as fake with the original
// device. This handles the case where captured real tensors (e.g. nn.Parameters,
// closed-over variables) are mixed with fake tensors in an op call.
static at::Tensor real_tensor_to_fake(
    const at::Tensor& t,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  auto original_device = t.device();
  at::Tensor meta_t;
  {
    // Exclude Fake key so empty_strided doesn't re-enter fakeFallback.
    c10::impl::ExcludeDispatchKeyGuard guard(
        c10::DispatchKeySet(c10::DispatchKey::Fake));
    meta_t = at::empty_strided(
        t.sizes(),
        t.strides(),
        t.options().device(c10::DeviceType::Meta));
  }
  if (t.requires_grad()) {
    meta_t.set_requires_grad(true);
  }
  transmute_to_fake(meta_t, original_device, mode);
  return meta_t;
}

static void convert_non_fake_inputs(
    torch::jit::Stack* stack,
    size_t arguments_begin,
    size_t num_arguments,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  for (size_t idx = 0; idx < num_arguments; ++idx) {
    auto& ivalue = (*stack)[arguments_begin + idx];
    if (ivalue.isTensor()) {
      const auto& t = ivalue.toTensor();
      if (t.defined() && !t.is_fake()) {
        (*stack)[arguments_begin + idx] = real_tensor_to_fake(t, mode);
      }
    } else if (ivalue.isTensorList()) {
      auto tensors = ivalue.toTensorList();
      bool needs_convert = false;
      for (const auto i : c10::irange(tensors.size())) {
        at::Tensor t = tensors[i];
        if (t.defined() && !t.is_fake()) {
          needs_convert = true;
          break;
        }
      }
      if (needs_convert) {
        auto new_list = c10::List<at::Tensor>();
        for (const auto i : c10::irange(tensors.size())) {
          at::Tensor t = tensors[i];
          if (t.defined() && !t.is_fake()) {
            new_list.push_back(real_tensor_to_fake(t, mode));
          } else {
            new_list.push_back(t);
          }
        }
        (*stack)[arguments_begin + idx] = new_list;
      }
    } else if (ivalue.isOptionalTensorList()) {
      auto opt_tensors = ivalue.toOptionalTensorList();
      bool needs_convert = false;
      for (const auto i : c10::irange(opt_tensors.size())) {
        std::optional<at::Tensor> ot = opt_tensors[i];
        if (ot.has_value() && ot->defined() && !ot->is_fake()) {
          needs_convert = true;
          break;
        }
      }
      if (needs_convert) {
        auto new_list = c10::List<std::optional<at::Tensor>>();
        for (const auto i : c10::irange(opt_tensors.size())) {
          std::optional<at::Tensor> ot = opt_tensors[i];
          if (ot.has_value() && ot->defined() && !ot->is_fake()) {
            new_list.push_back(real_tensor_to_fake(*ot, mode));
          } else {
            new_list.push_back(ot);
          }
        }
        (*stack)[arguments_begin + idx] = new_list;
      }
    }
  }
}

static void wrap_outputs(
    torch::jit::Stack* stack,
    size_t returns_begin,
    size_t num_returns,
    c10::Device fake_device,
    const std::shared_ptr<c10::FakeTensorMode>& mode) {
  auto returns = torch::jit::last(*stack, num_returns);
  for (size_t idx = 0; idx < num_returns; ++idx) {
    const auto& ivalue = returns[idx];
    if (ivalue.isTensor()) {
      const auto& t = ivalue.toTensor();
      if (needs_transmute(t)) {
        transmute_to_fake(t, fake_device, mode);
      }
    } else if (ivalue.isTensorList()) {
      auto tensors = ivalue.toTensorList();
      for (const auto i : c10::irange(tensors.size())) {
        at::Tensor t = tensors[i];
        if (needs_transmute(t)) {
          transmute_to_fake(t, fake_device, mode);
        }
      }
    } else if (ivalue.isOptionalTensorList()) {
      auto opt_tensors = ivalue.toOptionalTensorList();
      for (const auto i : c10::irange(opt_tensors.size())) {
        std::optional<at::Tensor> ot = opt_tensors[i];
        if (ot.has_value() && needs_transmute(*ot)) {
          transmute_to_fake(*ot, fake_device, mode);
        }
      }
    }
  }
}

void fakeFallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatchKeySet,
    torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  const auto arguments_begin = stack->size() - num_arguments;

  auto mode = c10::impl::FakeTensorModeTLS::get_state();

  // Convert any real tensor inputs to C++ fake tensors. This handles
  // captured variables, nn.Parameters, and inline tensor constants that
  // aren't already fake.
  convert_non_fake_inputs(stack, arguments_begin, num_arguments, mode);

  // If this op has no computed Meta kernel, fall back to Python handling
  // (decompositions, fake_impls, etc.) via the stored bridge mode.
  if (!op.hasComputedKernelForDispatchKey(c10::DispatchKey::Meta)) {
    if (mode && mode->python_fallback_mode_) {
      c10::impl::TorchDispatchModeTLS::push_non_infra_mode_onto_stack(
          mode->python_fallback_mode_);
      c10::impl::IncludeDispatchKeyGuard python_guard(c10::DispatchKey::Python);
      auto ks = dispatchKeySet.remove(c10::DispatchKey::Fake) |
          c10::DispatchKeySet(c10::DispatchKey::Python);
      op.redispatchBoxed(ks, stack);
      c10::impl::TorchDispatchModeTLS::pop_stack();
      return;
    }
  }

  auto fake_device = get_common_device(stack, num_arguments);

  if (!fake_device.has_value()) {
    fake_device = rewrite_device_args_to_meta(
        stack, arguments_begin, num_arguments, schema);
    if (!fake_device.has_value()) {
      fake_device = c10::Device(c10::DeviceType::CPU);
    }
  }

  {
    c10::impl::ExcludeDispatchKeyGuard guard(
        c10::DispatchKeySet(c10::DispatchKey::Fake) |
        c10::DispatchKeySet(c10::DispatchKey::Python) |
        c10::DispatchKeySet(c10::DispatchKey::PythonTLSSnapshot));
    c10::impl::IncludeDispatchKeyGuard meta_guard(c10::DispatchKey::Meta);
    auto ks = dispatchKeySet.remove(c10::DispatchKey::Fake) |
        c10::DispatchKeySet(c10::DispatchKey::Meta);
    op.redispatchBoxed(ks, stack);
  }

  const auto num_returns = schema.returns().size();
  const auto returns_begin = stack->size() - num_returns;
  wrap_outputs(stack, returns_begin, num_returns, *fake_device, mode);
}

TORCH_LIBRARY_IMPL(_, Fake, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fakeFallback>());
}

} // anonymous namespace
