import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np
from test_mps import TestCaseMPS

import torch
from torch.testing._internal.common_utils import IS_CI, NoTest, run_tests, TestCase


if torch.backends.mps.is_available():
    from torch.mps import load_safetensors as mps_load_safetensors
else:
    print("MPS not available, skipping MPSBulkLoader tests", file=sys.stderr)
    TestCase = NoTest


class TestMPSBulkLoaderFileGeneration:
    """Helper class for generating test safetensors files"""

    @staticmethod
    def create_safetensors_file(filename, tensor_specs, metadata=None):
        """
        Create a safetensors file for testing.

        Args:
            filename: Path to output file
            tensor_specs: List of (name, data, dtype, shape) tuples
            metadata: Optional metadata dict
        """
        if metadata is None:
            metadata = {}

        header = {"__metadata__": metadata}
        current_offset = 0

        tensor_data = b""

        for name, data, dtype, shape in tensor_specs:
            if isinstance(data, (list, tuple)):
                if len(data) == 0:
                    tensor_bytes = b""
                else:
                    tensor_bytes = torch.tensor(data, dtype=dtype).numpy().tobytes()
            elif isinstance(data, torch.Tensor):
                tensor_bytes = data.to(dtype).numpy().tobytes()
            elif isinstance(data, np.ndarray):
                tensor_bytes = data.astype(
                    TestMPSBulkLoaderFileGeneration._torch_to_numpy_dtype(dtype)
                ).tobytes()
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

            if len(tensor_bytes) == 0:
                end_offset = current_offset
            else:
                end_offset = current_offset + len(tensor_bytes)

            header[name] = {
                "dtype": TestMPSBulkLoaderFileGeneration._torch_to_safetensors_dtype(
                    dtype
                ),
                "shape": list(shape),
                "data_offsets": [current_offset, end_offset],
            }

            tensor_data += tensor_bytes
            current_offset += len(tensor_bytes)

        with open(filename, "wb") as f:
            header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(tensor_data)

    @staticmethod
    def create_corrupted_file(filename, corruption_type):
        """Create various types of corrupted safetensors files for testing"""
        if corruption_type == "truncated_header_size":
            with open(filename, "wb") as f:
                f.write(b"\x00\x00\x00")

        elif corruption_type == "zero_header_size":
            with open(filename, "wb") as f:
                f.write(struct.pack("<Q", 0))

        elif corruption_type == "excessive_header_size":
            with open(filename, "wb") as f:
                f.write(struct.pack("<Q", 200 * 1024 * 1024))

        elif corruption_type == "malformed_json":
            with open(filename, "wb") as f:
                bad_json = b'{"tensor1": {"dtype": "F32", "shape": [2, 2], '
                f.write(struct.pack("<Q", len(bad_json)))
                f.write(bad_json)

        elif corruption_type == "missing_dtype":
            header = {"tensor1": {"shape": [2, 2], "data_offsets": [0, 16]}}
            header_json = json.dumps(header).encode("utf-8")
            with open(filename, "wb") as f:
                f.write(struct.pack("<Q", len(header_json)))
                f.write(header_json)
                f.write(b"\x00" * 16)

        elif corruption_type == "invalid_offset_range":
            header = {
                "tensor1": {"dtype": "F32", "shape": [2, 2], "data_offsets": [16, 8]}
            }
            header_json = json.dumps(header).encode("utf-8")
            with open(filename, "wb") as f:
                f.write(struct.pack("<Q", len(header_json)))
                f.write(header_json)
                f.write(b"\x00" * 16)

        elif corruption_type == "negative_dimensions":
            header = {
                "tensor1": {"dtype": "F32", "shape": [2, -1], "data_offsets": [0, 8]}
            }
            header_json = json.dumps(header).encode("utf-8")
            with open(filename, "wb") as f:
                f.write(struct.pack("<Q", len(header_json)))
                f.write(header_json)
                f.write(b"\x00" * 8)

        elif corruption_type == "size_max_overflow":
            header = {
                "tensor1": {
                    "dtype": "F32",
                    "shape": [2, 2],
                    "data_offsets": [sys.maxsize - 8, sys.maxsize + 8],
                }
            }
            header_json = json.dumps(header).encode("utf-8")
            with open(filename, "wb") as f:
                f.write(struct.pack("<Q", len(header_json)))
                f.write(header_json)
                f.write(b"\x00" * 16)

        elif corruption_type == "truncated_tensor_data":
            header = {
                "tensor1": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}
            }
            header_json = json.dumps(header).encode("utf-8")
            with open(filename, "wb") as f:
                f.write(struct.pack("<Q", len(header_json)))
                f.write(header_json)
                f.write(b"\x00" * 8)

    @staticmethod
    def _torch_to_safetensors_dtype(torch_dtype):
        """Convert torch dtype to safetensors dtype string"""
        mapping = {
            torch.float64: "F64",
            torch.float32: "F32",
            torch.float16: "F16",
            torch.bfloat16: "BF16",
            torch.int64: "I64",
            torch.int32: "I32",
            torch.int16: "I16",
            torch.int8: "I8",
            torch.uint8: "U8",
            torch.bool: "BOOL",
        }
        return mapping[torch_dtype]

    @staticmethod
    def _torch_to_numpy_dtype(torch_dtype):
        """Convert torch dtype to numpy dtype"""
        mapping = {
            torch.float64: np.float64,
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.int64: np.int64,
            torch.int32: np.int32,
            torch.int16: np.int16,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
        }
        return mapping[torch_dtype]


class TestMPSBulkLoader(TestCaseMPS):
    """Comprehensive test suite for MPSBulkLoader functionality"""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.cleanup_files = []

    def tearDown(self):
        for filepath in self.cleanup_files:
            try:
                os.remove(filepath)
            except FileNotFoundError:
                pass
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        super().tearDown()

    def _get_temp_filename(self, suffix=".safetensors"):
        """Generate a temporary filename"""
        import uuid

        filename = os.path.join(self.temp_dir, f"test_{uuid.uuid4().hex}{suffix}")
        self.cleanup_files.append(filename)
        return filename

    # ============================================================================
    # Constructor/Destructor Tests
    # ============================================================================

    def test_constructor_valid_file(self):
        """Test MPSBulkLoader constructor with valid file"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(
            filename, [("tensor1", [1.0, 2.0, 3.0, 4.0], torch.float32, [2, 2])]
        )

        tensors = mps_load_safetensors(filename)
        self.assertEqual(len(tensors), 1)
        self.assertTrue("tensor1" in tensors)
        self.assertEqual(tensors["tensor1"].shape, (2, 2))
        self.assertTrue(tensors["tensor1"].is_mps)

    def test_constructor_nonexistent_file(self):
        """Test MPSBulkLoader constructor with nonexistent file"""
        with self.assertRaises(RuntimeError):
            mps_load_safetensors("/nonexistent/path/file.safetensors")

    def test_constructor_permission_denied(self):
        """Test MPSBulkLoader constructor with permission denied"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(
            filename, [("tensor1", [1.0], torch.float32, [1])]
        )

        os.chmod(filename, 0o000)
        try:
            with self.assertRaises(RuntimeError):
                mps_load_safetensors(filename)
        finally:
            os.chmod(filename, 0o644)

    # ============================================================================
    # Data Type Conversion Tests
    # ============================================================================

    def test_dtype_conversion_all_valid_types(self):
        """Test all supported dtype conversions"""
        test_dtypes = [
            (torch.float32, "F32", [1.0, 2.0]),
            (torch.float16, "F16", [1.0, 2.0]),
            (torch.int64, "I64", [1, 2]),
            (torch.int32, "I32", [1, 2]),
            (torch.int16, "I16", [1, 2]),
            (torch.int8, "I8", [1, 2]),
            (torch.uint8, "U8", [1, 2]),
            (torch.bool, "BOOL", [True, False]),
        ]

        for torch_dtype, safetensors_dtype, test_data in test_dtypes:
            with self.subTest(dtype=torch_dtype):
                filename = self._get_temp_filename()
                TestMPSBulkLoaderFileGeneration.create_safetensors_file(
                    filename, [("test_tensor", test_data, torch_dtype, [2])]
                )

                tensors = mps_load_safetensors(filename)
                self.assertEqual(tensors["test_tensor"].dtype, torch_dtype)
                self.assertEqual(tensors["test_tensor"].shape, (2,))
                self.assertTrue(tensors["test_tensor"].is_mps)

    # ============================================================================
    # Header Parsing Tests
    # ============================================================================

    def test_parse_header_simple_valid(self):
        """Test parsing a simple valid header"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(
            filename,
            [
                ("weight", [1.0, 2.0, 3.0, 4.0], torch.float32, [2, 2]),
                ("bias", [0.1, 0.2, 0.3], torch.float32, [3]),
            ],
        )

        tensors = mps_load_safetensors(filename)
        self.assertEqual(len(tensors), 2)
        self.assertTrue("weight" in tensors)
        self.assertTrue("bias" in tensors)
        self.assertEqual(tensors["weight"].shape, (2, 2))
        self.assertEqual(tensors["bias"].shape, (3,))

    def test_parse_header_empty_file(self):
        """Test parsing header of empty tensor file"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(filename, [])

        tensors = mps_load_safetensors(filename)
        self.assertEqual(len(tensors), 0)

    def test_parse_header_with_metadata(self):
        """Test parsing header with metadata"""
        filename = self._get_temp_filename()
        metadata = {"version": "1.0", "author": "test"}
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(
            filename, [("tensor1", [1.0, 2.0], torch.float32, [2])], metadata=metadata
        )

        tensors = mps_load_safetensors(filename)
        self.assertEqual(len(tensors), 1)
        self.assertTrue("tensor1" in tensors)

    def test_parse_header_large_file_many_tensors(self):
        """Test parsing header with many tensors (stress test)"""
        filename = self._get_temp_filename()
        tensor_specs = []
        for i in range(100):
            tensor_specs.append(
                (f"tensor_{i}", [float(i), float(i + 1)], torch.float32, [2])
            )

        TestMPSBulkLoaderFileGeneration.create_safetensors_file(filename, tensor_specs)

        tensors = mps_load_safetensors(filename)
        self.assertEqual(len(tensors), 100)
        for i in range(100):
            self.assertTrue(f"tensor_{i}" in tensors)
            self.assertEqual(tensors[f"tensor_{i}"].shape, (2,))

    def test_parse_header_truncated_header_size(self):
        """Test parsing with truncated header size"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_corrupted_file(
            filename, "truncated_header_size"
        )

        with self.assertRaises(RuntimeError):
            mps_load_safetensors(filename)

    def test_parse_header_zero_header_size(self):
        """Test parsing with zero header size"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_corrupted_file(
            filename, "zero_header_size"
        )

        with self.assertRaises(RuntimeError):
            mps_load_safetensors(filename)

    def test_parse_header_excessive_header_size(self):
        """Test parsing with excessively large header size"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_corrupted_file(
            filename, "excessive_header_size"
        )

        with self.assertRaises(RuntimeError):
            mps_load_safetensors(filename)

    def test_parse_header_malformed_json(self):
        """Test parsing with malformed JSON"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_corrupted_file(
            filename, "malformed_json"
        )

        with self.assertRaises(RuntimeError):
            mps_load_safetensors(filename)

    def test_parse_header_missing_required_fields(self):
        """Test parsing with missing required fields"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_corrupted_file(filename, "missing_dtype")

        with self.assertRaises(RuntimeError):
            mps_load_safetensors(filename)

    def test_parse_header_invalid_offsets(self):
        """Test parsing with invalid offset ranges"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_corrupted_file(
            filename, "invalid_offset_range"
        )

        with self.assertRaises(RuntimeError):
            mps_load_safetensors(filename)

    def test_parse_header_negative_dimensions(self):
        """Test parsing with negative dimensions"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_corrupted_file(
            filename, "negative_dimensions"
        )

        with self.assertRaises(RuntimeError):
            mps_load_safetensors(filename)

    def test_parse_header_offset_overflow(self):
        """Test parsing with offset overflow"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_corrupted_file(
            filename, "size_max_overflow"
        )

        with self.assertRaises(RuntimeError):
            mps_load_safetensors(filename)

    # ============================================================================
    # Parallel Read Tests
    # ============================================================================

    def test_parallel_read_single_tensor(self):
        """Test parallel reading of single tensor"""
        filename = self._get_temp_filename()
        test_data = [1.0, 2.0, 3.0, 4.0]
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(
            filename, [("weight", test_data, torch.float32, [2, 2])]
        )

        tensors = mps_load_safetensors(filename)
        result = tensors["weight"].cpu().numpy()
        expected = np.array(test_data, dtype=np.float32).reshape(2, 2)
        np.testing.assert_array_equal(result, expected)

    def test_parallel_read_multiple_tensors(self):
        """Test parallel reading of multiple tensors"""
        filename = self._get_temp_filename()
        tensor_specs = [
            ("tensor1", [1.0, 2.0], torch.float32, [2]),
            ("tensor2", [10, 20, 30], torch.int32, [3]),
            ("tensor3", [True, False], torch.bool, [2]),
        ]
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(filename, tensor_specs)

        tensors = mps_load_safetensors(filename)

        result1 = tensors["tensor1"].cpu().numpy()
        expected1 = np.array([1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(result1, expected1)

        result2 = tensors["tensor2"].cpu().numpy()
        expected2 = np.array([10, 20, 30], dtype=np.int32)
        np.testing.assert_array_equal(result2, expected2)

        result3 = tensors["tensor3"].cpu().numpy()
        expected3 = np.array([True, False], dtype=bool)
        np.testing.assert_array_equal(result3, expected3)

    def test_parallel_read_empty_list(self):
        """Test parallel reading with empty tensor list"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(filename, [])

        tensors = mps_load_safetensors(filename)
        self.assertEqual(len(tensors), 0)

    def test_parallel_read_high_concurrency(self):
        """Test parallel reading with many tensors (stress test)"""
        filename = self._get_temp_filename()
        tensor_specs = []
        expected_data = {}

        for i in range(50):
            data = [float(j + i * 10) for j in range(10)]
            tensor_specs.append((f"tensor_{i}", data, torch.float32, [10]))
            expected_data[f"tensor_{i}"] = np.array(data, dtype=np.float32)
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(filename, tensor_specs)

        tensors = mps_load_safetensors(filename)
        self.assertEqual(len(tensors), 50)

        for i in range(50):
            tensor_name = f"tensor_{i}"
            self.assertTrue(tensor_name in tensors)
            result = tensors[tensor_name].cpu().numpy()
            np.testing.assert_array_equal(result, expected_data[tensor_name])

    def test_parallel_read_truncated_file(self):
        """Test parallel reading with truncated file data"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_corrupted_file(
            filename, "truncated_tensor_data"
        )

        with self.assertRaises(RuntimeError):
            mps_load_safetensors(filename)

    # ============================================================================
    # Data Integrity and Correctness Tests
    # ============================================================================

    def test_data_integrity_all_dtypes(self):
        """Test data integrity for all supported dtypes"""
        test_cases = [
            (torch.float32, [1.5, -2.5, 3.14159, 0.0], [4]),
            (torch.float16, [1.0, -1.0, 2.0], [3]),
            (torch.int64, [-9223372036854775808, 0, 9223372036854775807], [3]),
            (torch.int32, [-2147483648, 0, 2147483647], [3]),
            (torch.int16, [-32768, 0, 32767], [3]),
            (torch.int8, [-128, 0, 127], [3]),
            (torch.uint8, [0, 128, 255], [3]),
            (torch.bool, [True, False, True, False], [2, 2]),
        ]

        for dtype, test_data, shape in test_cases:
            with self.subTest(dtype=dtype):
                filename = self._get_temp_filename()
                TestMPSBulkLoaderFileGeneration.create_safetensors_file(
                    filename, [("test_tensor", test_data, dtype, shape)]
                )

                tensors = mps_load_safetensors(filename)
                result = tensors["test_tensor"].cpu()

                expected = torch.tensor(test_data, dtype=dtype).reshape(shape)
                self.assertEqual(result, expected)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.shape, tuple(shape))

    def test_data_integrity_large_tensors(self):
        """Test data integrity with larger tensors"""
        filename = self._get_temp_filename()

        size = 1000
        test_data = [float(i * 2 + 1) for i in range(size)]
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(
            filename, [("large_tensor", test_data, torch.float32, [size])]
        )

        tensors = mps_load_safetensors(filename)
        result = tensors["large_tensor"].cpu().numpy()
        expected = np.array(test_data, dtype=np.float32)

        np.testing.assert_array_equal(result, expected)

    def test_data_integrity_multidimensional(self):
        """Test data integrity with multidimensional tensors"""
        filename = self._get_temp_filename()

        test_data = list(range(24))
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(
            filename, [("tensor_3d", test_data, torch.int32, [2, 3, 4])]
        )

        tensors = mps_load_safetensors(filename)
        result = tensors["tensor_3d"].cpu()
        expected = torch.tensor(test_data, dtype=torch.int32).reshape(2, 3, 4)

        self.assertEqual(result, expected)
        self.assertEqual(result.shape, (2, 3, 4))

    # ============================================================================
    # Memory Management Tests
    # ============================================================================

    def test_mps_tensor_allocation(self):
        """Test that tensors are properly allocated on MPS device"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(
            filename, [("test_tensor", [1.0, 2.0, 3.0, 4.0], torch.float32, [2, 2])]
        )

        tensors = mps_load_safetensors(filename)
        tensor = tensors["test_tensor"]

        self.assertTrue(tensor.is_mps)
        self.assertEqual(tensor.device.type, "mps")
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.shape, (2, 2))

        result = tensor + 1.0
        self.assertTrue(result.is_mps)

    def test_multiple_tensor_mps_allocation(self):
        """Test MPS allocation for multiple tensors"""
        filename = self._get_temp_filename()
        tensor_specs = [
            ("tensor1", [1.0, 2.0], torch.float32, [2]),
            ("tensor2", [10, 20, 30], torch.int32, [3]),
            ("tensor3", [True, False, True], torch.bool, [3]),
        ]
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(filename, tensor_specs)

        tensors = mps_load_safetensors(filename)

        for name, tensor in tensors.items():
            self.assertTrue(tensor.is_mps, f"Tensor {name} not on MPS device")
            self.assertEqual(tensor.device.type, "mps")

    # ============================================================================
    # Integration and End-to-End Tests
    # ============================================================================

    def test_complete_workflow_simple_model(self):
        """Test complete workflow with simple model-like data"""
        filename = self._get_temp_filename()
        tensor_specs = [
            ("model.weight", [[1.0, 2.0], [3.0, 4.0]], torch.float32, [2, 2]),
            ("model.bias", [0.1, 0.2], torch.float32, [2]),
            ("classifier.weight", [[0.5, 0.6, 0.7, 0.8]], torch.float32, [1, 4]),
        ]
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(filename, tensor_specs)

        tensors = mps_load_safetensors(filename)

        expected_names = {"model.weight", "model.bias", "classifier.weight"}
        self.assertEqual(set(tensors.keys()), expected_names)

        self.assertEqual(tensors["model.weight"].shape, (2, 2))
        self.assertEqual(tensors["model.bias"].shape, (2,))
        self.assertEqual(tensors["classifier.weight"].shape, (1, 4))

        for tensor in tensors.values():
            self.assertTrue(tensor.is_mps)

    def test_complete_workflow_mixed_dtypes(self):
        """Test complete workflow with mixed data types"""
        filename = self._get_temp_filename()
        tensor_specs = [
            ("float_tensor", [1.5, 2.5], torch.float32, [2]),
            ("int_tensor", [10, 20], torch.int64, [2]),
            ("bool_tensor", [True, False], torch.bool, [2]),
            ("half_tensor", [1.0, 2.0], torch.float16, [2]),
        ]
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(filename, tensor_specs)

        tensors = mps_load_safetensors(filename)

        self.assertEqual(tensors["float_tensor"].dtype, torch.float32)
        self.assertEqual(tensors["int_tensor"].dtype, torch.int64)
        self.assertEqual(tensors["bool_tensor"].dtype, torch.bool)
        self.assertEqual(tensors["half_tensor"].dtype, torch.float16)

        for tensor in tensors.values():
            self.assertTrue(tensor.is_mps)

    # ============================================================================
    # Error Handling and Edge Cases
    # ============================================================================

    def test_error_handling_invalid_dtype_string(self):
        """Test error handling for invalid dtype strings"""
        filename = self._get_temp_filename()

        header = {
            "tensor1": {"dtype": "INVALID_TYPE", "shape": [2], "data_offsets": [0, 8]}
        }
        header_json = json.dumps(header).encode("utf-8")

        with open(filename, "wb") as f:
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(b"\x00" * 8)

        with self.assertRaises(RuntimeError):
            mps_load_safetensors(filename)

    def test_error_handling_malformed_offsets(self):
        """Test error handling for malformed offset arrays"""
        filename = self._get_temp_filename()

        header = {"tensor1": {"dtype": "F32", "shape": [2], "data_offsets": [0]}}
        header_json = json.dumps(header).encode("utf-8")

        with open(filename, "wb") as f:
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(b"\x00" * 8)

        with self.assertRaises(RuntimeError):
            mps_load_safetensors(filename)

    def test_edge_case_zero_sized_tensors(self):
        """Test handling of zero-sized tensors"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(
            filename,
            [("empty_tensor", torch.empty(0, dtype=torch.float32), torch.float32, [0])],
        )

        tensors = mps_load_safetensors(filename)
        tensor = tensors["empty_tensor"]

        self.assertEqual(tensor.shape, (0,))
        self.assertTrue(tensor.is_mps)
        self.assertEqual(tensor.numel(), 0)

    def test_edge_case_scalar_tensors(self):
        """Test handling of scalar tensors"""
        filename = self._get_temp_filename()
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(
            filename, [("scalar_tensor", [42.0], torch.float32, [1])]
        )

        tensors = mps_load_safetensors(filename)
        tensor = tensors["scalar_tensor"]

        self.assertEqual(tensor.shape, (1,))
        self.assertTrue(tensor.is_mps)
        self.assertEqual(tensor.item(), 42.0)

    # ============================================================================
    # Performance and Stress Tests
    # ============================================================================

    @unittest.skipIf(IS_CI, "Skip performance test in CI")
    def test_performance_large_number_of_tensors(self):
        """Performance test with large number of tensors"""
        filename = self._get_temp_filename()

        tensor_specs = []
        num_tensors = 500
        for i in range(num_tensors):
            tensor_specs.append((f"tensor_{i}", [float(i)], torch.float32, [1]))

        TestMPSBulkLoaderFileGeneration.create_safetensors_file(filename, tensor_specs)

        import time

        start_time = time.time()
        tensors = mps_load_safetensors(filename)
        end_time = time.time()

        self.assertEqual(len(tensors), num_tensors)

        load_time = end_time - start_time
        print(f"\nLoaded {num_tensors} tensors in {load_time:.3f} seconds")
        print(f"Average per tensor: {load_time / num_tensors * 1000:.2f} ms")

    @unittest.skipIf(IS_CI, "Skip performance test in CI")
    def test_performance_large_tensor_data(self):
        """Performance test with large tensor data"""
        filename = self._get_temp_filename()

        size = 100000
        large_data = [float(i) for i in range(size)]
        TestMPSBulkLoaderFileGeneration.create_safetensors_file(
            filename, [("large_tensor", large_data, torch.float32, [size])]
        )

        import time

        start_time = time.time()
        tensors = mps_load_safetensors(filename)
        end_time = time.time()

        self.assertEqual(tensors["large_tensor"].numel(), size)
        self.assertTrue(tensors["large_tensor"].is_mps)

        load_time = end_time - start_time
        data_size_mb = size * 4 / (1024 * 1024)
        print(f"\nLoaded {data_size_mb:.1f} MB tensor in {load_time:.3f} seconds")
        print(f"Throughput: {data_size_mb / load_time:.1f} MB/s")

    # ============================================================================
    # Regression Tests for Specific Issues
    # ============================================================================

    def test_regression_concurrent_error_handling(self):
        """Regression test for concurrent error handling race conditions"""
        filename = self._get_temp_filename()

        tensor_specs = []
        for i in range(10):
            tensor_specs.append((f"tensor_{i}", [1.0], torch.float32, [100]))

        header = {}
        offset = 0
        for i in range(10):
            header[f"tensor_{i}"] = {
                "dtype": "F32",
                "shape": [100],
                "data_offsets": [offset, offset + 400],
            }
            offset += 400

        header_json = json.dumps(header).encode("utf-8")
        with open(filename, "wb") as f:
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(b"\x00" * 40)

        with self.assertRaises(RuntimeError) as cm:
            mps_load_safetensors(filename)

        error_msg = str(cm.exception)
        self.assertIn("Failed to read tensor", error_msg)


class TestMPSBulkLoaderCompatibility(TestCaseMPS):
    """Tests for compatibility with standard safetensors format"""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.cleanup_files = []

    def tearDown(self):
        for filepath in self.cleanup_files:
            try:
                os.remove(filepath)
            except FileNotFoundError:
                pass
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass
        super().tearDown()

    def _get_temp_filename(self, suffix=".safetensors"):
        import uuid

        filename = os.path.join(self.temp_dir, f"test_{uuid.uuid4().hex}{suffix}")
        self.cleanup_files.append(filename)
        return filename

    def test_compatibility_with_torch_save_format(self):
        """Test compatibility with tensors saved by standard PyTorch methods"""
        try:
            from safetensors.torch import load_file, save_file
        except ImportError:
            self.skipTest("safetensors library not available for compatibility test")

        tensors = {
            "weight": torch.randn(10, 5, dtype=torch.float32),
            "bias": torch.randn(5, dtype=torch.float32),
            "scale": torch.tensor([1.5], dtype=torch.float32),
        }

        filename = self._get_temp_filename()

        save_file(tensors, filename)

        mps_tensors = mps_load_safetensors(filename)

        self.assertEqual(len(mps_tensors), len(tensors))
        for name, original_tensor in tensors.items():
            self.assertTrue(name in mps_tensors)
            loaded_tensor = mps_tensors[name].cpu()
            self.assertEqual(loaded_tensor, original_tensor)
            self.assertEqual(loaded_tensor.shape, original_tensor.shape)
            self.assertEqual(loaded_tensor.dtype, original_tensor.dtype)

        for tensor in mps_tensors.values():
            self.assertTrue(tensor.is_mps)


if __name__ == "__main__":
    run_tests()
