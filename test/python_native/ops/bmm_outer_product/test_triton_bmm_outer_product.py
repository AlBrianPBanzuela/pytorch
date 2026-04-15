# Owner(s): ["module: dsl-native-ops"]
"""
Tests for Triton BMM Outer Product implementation using BinaryUfuncInfo with DSL support.

This module provides OpInfo definitions for testing the Triton-optimized
batch matrix multiplication implementation specifically for outer product cases.
"""

import unittest
from functools import partial

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    onlyCUDA,
    skipCUDAIf,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_dtype import _dispatch_dtypes, floating_types
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.opinfo.core import (
    BinaryUfuncInfo,
    DecorateInfo,
    SampleInput,
)


def sample_inputs_triton_bmm_outer_product(
    op_info, device, dtype, requires_grad, **kwargs
):
    """
    Generate sample inputs for Triton BMM outer product testing.

    This generates inputs that match the outer product pattern:
    - a: shape (B, M, 1) - batch of column vectors
    - b: shape (B, 1, N) - batch of row vectors
    - output: (B, M, N) - batch of outer products
    """
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # Test various batch sizes and dimensions
    test_configs = [
        (2, 16, 32),  # Small: B=2, M=16, N=32
        (4, 64, 64),  # Medium: B=4, M=64, N=64
        (8, 128, 256),  # Large: B=8, M=128, N=256
        (1, 512, 128),  # Single batch, large dims
        (16, 32, 64),  # Many batches
        (3, 7, 11),  # Non-power-of-2 dimensions
    ]

    for batch_size, m_dim, n_dim in test_configs:
        # Create tensors with outer product pattern
        # a: (B, M, 1) - batch of column vectors
        a = make_arg((batch_size, m_dim, 1)).contiguous()

        # b: (B, 1, N) - batch of row vectors
        b = make_arg((batch_size, 1, n_dim)).contiguous()

        yield SampleInput(a, args=(b,))

        # Test with different value ranges
        yield SampleInput(
            make_arg((batch_size, m_dim, 1), low=-2.0, high=2.0).contiguous(),
            args=(make_arg((batch_size, 1, n_dim), low=-2.0, high=2.0).contiguous(),),
        )

        # Test with small positive values
        yield SampleInput(
            make_arg((batch_size, m_dim, 1), low=0.1, high=1.0).contiguous(),
            args=(make_arg((batch_size, 1, n_dim), low=0.1, high=1.0).contiguous(),),
        )


def reference_bmm_outer_product(a, b):
    """Reference implementation using standard torch.bmm."""
    return torch.bmm(a, b)


# Import the Triton kernel and availability check
try:
    from torch._native.ops.bmm_outer_product.triton_kernels import (
        bmm_outer_product as triton_bmm_outer_product_kernel,
    )

    HAS_TRITON_BMM = True
except ImportError:
    # Create a dummy function if Triton is not available
    def triton_bmm_outer_product_kernel(a, b):
        return torch.bmm(a, b)

    HAS_TRITON_BMM = False


def verify_outer_product_conditions(a, b):
    """
    Verify if tensors meet the conditions for Triton BMM outer product optimization.

    Returns True if Triton should be used, False otherwise.
    """
    return (
        a.ndim == 3
        and b.ndim == 3
        and a.shape[2] == 1  # a is (B, M, 1)
        and b.shape[1] == 1  # b is (B, 1, N)
        and a.numel() > 0
        and b.numel() > 0
        and not a.is_complex()
        and not b.is_complex()
        and a.is_cuda
        and b.is_cuda
        and a.is_contiguous()
        and b.is_contiguous()
    )


# BinaryUfuncInfo for Triton BMM Outer Product testing - DIRECT IMPLEMENTATION TESTING
triton_bmm_outer_product_opinfo = BinaryUfuncInfo(
    "triton_bmm_outer_product",
    op=triton_bmm_outer_product_kernel,  # Direct call to Triton kernel
    dsl_name="triton",  # DSL backend identification
    # BinaryUfuncInfo supports floating point types, exclude complex
    dtypes=_dispatch_dtypes(floating_types()),
    dtypesIfCUDA=_dispatch_dtypes(floating_types()),  # CUDA required for Triton
    # Input generation - only generate inputs that meet outer product conditions
    sample_inputs_func=sample_inputs_triton_bmm_outer_product,
    # Reference function for numerical testing
    ref=reference_bmm_outer_product,
    # Binary operation properties
    supports_autograd=False,  # Direct kernel testing, not autodiff integration
    supports_forward_ad=False,
    supports_fwgrad_bwgrad=False,
    # Direct kernel testing doesn't support out parameter
    supports_out=False,
    # No inplace variant for BMM
    inplace_variant=None,
    # Device and precision decorators
    decorators=[
        # Standard tolerances for floating point operations
        DecorateInfo(
            toleranceOverride(
                {
                    torch.float32: tol(atol=1e-5, rtol=1e-5),
                    torch.float64: tol(atol=1e-10, rtol=1e-10),
                    torch.float16: tol(atol=1e-3, rtol=1e-3),
                    torch.bfloat16: tol(atol=1e-2, rtol=1e-2),
                }
            ),
            device_type="cuda",
        ),
        # Only test on CUDA since Triton requires CUDA
        DecorateInfo(
            onlyCUDA,
        ),
    ],
    # Skip conditions
    skips=(
        # Skip if CUDA is not available (Triton requires CUDA)
        DecorateInfo(
            skipCUDAIf(not torch.cuda.is_available(), "CUDA not available"),
        ),
        # Skip if Triton BMM is not available
        DecorateInfo(
            unittest.skipIf(
                not HAS_TRITON_BMM, "Triton BMM outer product not available"
            ),
        ),
        # Skip complex dtypes as BMM outer product doesn't support them meaningfully
        DecorateInfo(
            unittest.skip("Complex dtypes not supported for BMM outer product"),
            dtypes=(torch.cfloat, torch.cdouble),
        ),
        # Skip tests that generate incompatible tensor shapes (not outer product pattern)
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_batch_vs_slicing",
        ),
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_broadcasting",
        ),
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_contig_size1_large_dim",
        ),
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_contig_size1",
        ),
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_contig_vs_every_other",
        ),
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_contig_vs_transposed",
        ),
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_non_contig_expand",
        ),
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_non_contig",
        ),
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_not_broadcastable",
        ),
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_reference_numerics_extremal_values",
        ),
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_reference_numerics_large_values",
        ),
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_reference_numerics_small_values",
        ),
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_reference_numerics",
        ),
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_scalar_support",
        ),
        DecorateInfo(
            unittest.skip("BMM outer product requires specific 3D tensor pattern"),
            "TestBinaryUfuncs",
            "test_type_promotion",
        ),
    ),
    # Additional metadata for Triton-specific testing
    variant_test_name="triton_optimized",
)


def test_bmm_outer_product_conditions(self):
    """
    Test that Triton optimization conditions are properly detected.
    This is a custom test method for validating outer product pattern detection.
    """
    if not torch.cuda.is_available():
        self.skipTest("CUDA not available")

    device = torch.device("cuda")
    dtype = torch.float32

    # Test valid outer product pattern
    a_valid = make_tensor((4, 32, 1), device=device, dtype=dtype)
    b_valid = make_tensor((4, 1, 64), device=device, dtype=dtype)

    self.assertTrue(verify_outer_product_conditions(a_valid, b_valid))

    # Test invalid patterns
    # Wrong last dimension for a
    a_invalid = make_tensor((4, 32, 2), device=device, dtype=dtype)
    self.assertFalse(verify_outer_product_conditions(a_invalid, b_valid))

    # Wrong middle dimension for b
    b_invalid = make_tensor((4, 2, 64), device=device, dtype=dtype)
    self.assertFalse(verify_outer_product_conditions(a_valid, b_invalid))

    # Different batch sizes
    a_diff_batch = make_tensor((3, 32, 1), device=device, dtype=dtype)
    self.assertFalse(verify_outer_product_conditions(a_diff_batch, b_valid))


def create_outer_product_tensors(
    batch_size, m_dim, n_dim, device="cuda", dtype=torch.float32, **kwargs
):
    """
    Create tensors that meet Triton BMM outer product optimization conditions.
    """
    a = make_tensor((batch_size, m_dim, 1), device=device, dtype=dtype, **kwargs)
    b = make_tensor((batch_size, 1, n_dim), device=device, dtype=dtype, **kwargs)
    return a.contiguous(), b.contiguous()


if __name__ == "__main__":
    run_tests()


# Export the OpInfo for import by common_methods_invocations.py
__all__ = [
    "triton_bmm_outer_product_opinfo",
    "sample_inputs_triton_bmm_outer_product",
    "reference_bmm_outer_product",
    "verify_outer_product_conditions",
    "create_outer_product_tensors",
]
