import contextlib

import torch
from torch._subclasses.fake_tensor import FakeTensorConverter
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


@contextlib.contextmanager
def cpp_fake_tensor_mode(shape_env=None):
    if shape_env is None:
        shape_env = ShapeEnv()
    converter = FakeTensorConverter()
    torch._C._create_and_enter_fake_tensor_mode(converter, shape_env)
    try:
        yield shape_env
    finally:
        torch._C._exit_fake_tensor_mode()


class TestFakeTensorCpp(TestCase):
    # ---- torch.add ----

    def test_add_concrete_shapes(self, device):
        with cpp_fake_tensor_mode():
            a = torch.randn(3, 4, device=device)
            b = torch.randn(3, 4, device=device)
            c = a + b
            self.assertTrue(torch._C._is_fake_tensor(c))
            self.assertEqual(c.shape, (3, 4))

    def test_add_broadcast(self, device):
        with cpp_fake_tensor_mode():
            a = torch.randn(3, 4, device=device)
            b = torch.randn(4, device=device)
            c = torch.add(a, b)
            self.assertTrue(torch._C._is_fake_tensor(c))
            self.assertEqual(c.shape, (3, 4))

    def test_add_scalar(self, device):
        with cpp_fake_tensor_mode():
            a = torch.randn(2, 3, device=device)
            c = torch.add(a, 1.0)
            self.assertTrue(torch._C._is_fake_tensor(c))
            self.assertEqual(c.shape, (2, 3))

    # ---- torch.mm ----

    def test_mm(self, device):
        with cpp_fake_tensor_mode():
            a = torch.randn(3, 4, device=device)
            b = torch.randn(4, 5, device=device)
            c = torch.mm(a, b)
            self.assertTrue(torch._C._is_fake_tensor(c))
            self.assertEqual(c.shape, (3, 5))

    def test_mm_square(self, device):
        with cpp_fake_tensor_mode():
            a = torch.randn(4, 4, device=device)
            b = torch.randn(4, 4, device=device)
            c = torch.mm(a, b)
            self.assertEqual(c.shape, (4, 4))

    # ---- torch.split ----

    def test_split_even(self, device):
        with cpp_fake_tensor_mode():
            x = torch.randn(6, 4, device=device)
            parts = torch.split(x, 2, dim=0)
            self.assertEqual(len(parts), 3)
            for part in parts:
                self.assertTrue(torch._C._is_fake_tensor(part))
                self.assertEqual(part.shape, (2, 4))

    def test_split_uneven(self, device):
        with cpp_fake_tensor_mode():
            x = torch.randn(7, 4, device=device)
            parts = torch.split(x, 3, dim=0)
            self.assertEqual(len(parts), 3)
            self.assertEqual(parts[0].shape, (3, 4))
            self.assertEqual(parts[1].shape, (3, 4))
            self.assertEqual(parts[2].shape, (1, 4))

    def test_split_sections(self, device):
        with cpp_fake_tensor_mode():
            x = torch.randn(10, 3, device=device)
            parts = torch.split(x, [2, 3, 5], dim=0)
            self.assertEqual(len(parts), 3)
            self.assertEqual(parts[0].shape, (2, 3))
            self.assertEqual(parts[1].shape, (3, 3))
            self.assertEqual(parts[2].shape, (5, 3))

    # ---- torch.cat ----

    def test_cat_dim0(self, device):
        with cpp_fake_tensor_mode():
            a = torch.randn(2, 4, device=device)
            b = torch.randn(3, 4, device=device)
            c = torch.cat([a, b], dim=0)
            self.assertTrue(torch._C._is_fake_tensor(c))
            self.assertEqual(c.shape, (5, 4))

    def test_cat_dim1(self, device):
        with cpp_fake_tensor_mode():
            a = torch.randn(3, 2, device=device)
            b = torch.randn(3, 5, device=device)
            c = torch.cat([a, b], dim=1)
            self.assertTrue(torch._C._is_fake_tensor(c))
            self.assertEqual(c.shape, (3, 7))

    def test_cat_multiple(self, device):
        with cpp_fake_tensor_mode():
            tensors = [torch.randn(2, 3, device=device) for _ in range(4)]
            c = torch.cat(tensors, dim=0)
            self.assertTrue(torch._C._is_fake_tensor(c))
            self.assertEqual(c.shape, (8, 3))

    # ---- Shape env / dynamic shapes ----

    def test_shape_env_basic(self, device):
        """Tensors created inside the mode have concrete int shapes."""
        with cpp_fake_tensor_mode():
            x = torch.randn(5, 8, device=device)
            self.assertEqual(x.shape[0], 5)
            self.assertEqual(x.shape[1], 8)
            # Shape arithmetic works
            self.assertEqual(x.shape[0] + x.shape[1], 13)
            self.assertEqual(x.shape[0] * 2, 10)

    def test_shape_env_equality(self, device):
        """Shape dimensions of same-shaped tensors are equal."""
        with cpp_fake_tensor_mode():
            x = torch.randn(5, 8, device=device)
            y = torch.randn(5, 8, device=device)
            self.assertEqual(x.shape[0], y.shape[0])
            self.assertEqual(x.shape[1], y.shape[1])

    def test_dynamic_shapes_through_ops(self, device):
        """Shapes propagate correctly through a chain of operations."""
        with cpp_fake_tensor_mode():
            a = torch.randn(4, 6, device=device)
            b = torch.randn(6, 3, device=device)
            c = torch.mm(a, b)
            self.assertEqual(c.shape, (4, 3))
            # Split the result and cat it back
            parts = torch.split(c, 2, dim=0)
            self.assertEqual(len(parts), 2)
            self.assertEqual(parts[0].shape, (2, 3))
            self.assertEqual(parts[1].shape, (2, 3))
            d = torch.cat(list(parts), dim=0)
            self.assertEqual(d.shape, (4, 3))

    def test_add_shapes_propagate(self, device):
        """Add with broadcast propagates shapes correctly."""
        with cpp_fake_tensor_mode():
            a = torch.randn(1, 5, device=device)
            b = torch.randn(3, 1, device=device)
            c = a + b
            self.assertEqual(c.shape, (3, 5))

    # ---- Symbolic shapes ----

    def test_symbolic_shapes_create_unbacked(self, device):
        """Create a tensor with a symbolic size using ShapeEnv."""
        shape_env = ShapeEnv()
        with cpp_fake_tensor_mode(shape_env):
            s0 = shape_env.create_unbacked_symint()
            torch._check(s0 >= 0)
            torch._check(s0 >= 2)
            torch._check(s0 <= 100)
            x = torch.empty(s0, 4, device=device)
            self.assertTrue(torch._C._is_fake_tensor(x))
            self.assertEqual(x.shape[1], 4)

    def test_symbolic_shapes_mm(self, device):
        """Matrix multiply with symbolic dimensions propagates shapes."""
        shape_env = ShapeEnv()
        with cpp_fake_tensor_mode(shape_env):
            s0 = shape_env.create_unbacked_symint()
            torch._check(s0 >= 0)
            torch._check(s0 >= 2)
            torch._check(s0 <= 100)
            s1 = shape_env.create_unbacked_symint()
            torch._check(s1 >= 0)
            torch._check(s1 >= 2)
            torch._check(s1 <= 100)
            a = torch.empty(s0, 4, device=device)
            b = torch.empty(4, s1, device=device)
            c = torch.mm(a, b)
            self.assertTrue(torch._C._is_fake_tensor(c))
            # Inner dimension (4) is concrete
            # Outer dimensions should match the symbolic inputs
            self.assertEqual(c.shape[0], s0)
            self.assertEqual(c.shape[1], s1)

    def test_symbolic_shapes_cat(self, device):
        """Cat along a dimension with symbolic sizes."""
        shape_env = ShapeEnv()
        with cpp_fake_tensor_mode(shape_env):
            s0 = shape_env.create_unbacked_symint()
            torch._check(s0 >= 0)
            torch._check(s0 >= 2)
            torch._check(s0 <= 100)
            s1 = shape_env.create_unbacked_symint()
            torch._check(s1 >= 0)
            torch._check(s1 >= 2)
            torch._check(s1 <= 100)
            a = torch.empty(s0, 4, device=device)
            b = torch.empty(s1, 4, device=device)
            c = torch.cat([a, b], dim=0)
            self.assertTrue(torch._C._is_fake_tensor(c))
            self.assertEqual(c.shape[0], s0 + s1)
            self.assertEqual(c.shape[1], 4)

    # doesn't pass right now because symint handling for composite explicit autograd
    # functions is not handled yet
    # def test_symbolic_shapes_split(self, device):
    #     """Split with concrete split_size on a symbolic dimension."""
    #     shape_env = ShapeEnv()
    #     with cpp_fake_tensor_mode(shape_env):
    #         s0 = shape_env.create_unbacked_symint()
    #         torch._check(s0 >= 0)
    #         torch._check(s0 >= 2)
    #         torch._check(s0 <= 100)
    #         x = torch.empty(6, s0, device=device)
    #         parts = torch.split(x, 2, dim=0)
    #         self.assertEqual(len(parts), 3)
    #         for part in parts:
    #             self.assertTrue(torch._C._is_fake_tensor(part))
    #             self.assertEqual(part.shape[0], 2)
    #             self.assertEqual(part.shape[1], s0)

    def test_symbolic_shapes_add(self, device):
        """Add two tensors with symbolic shapes."""
        shape_env = ShapeEnv()
        with cpp_fake_tensor_mode(shape_env):
            s0 = shape_env.create_unbacked_symint()
            torch._check(s0 >= 0)
            torch._check(s0 >= 2)
            torch._check(s0 <= 100)
            a = torch.empty(s0, 4, device=device)
            b = torch.empty(s0, 4, device=device)
            c = a + b
            self.assertTrue(torch._C._is_fake_tensor(c))
            self.assertEqual(c.shape[0], s0)
            self.assertEqual(c.shape[1], 4)

    # ---- Context manager behavior ----

    def test_mode_does_not_leak(self, device):
        """After exiting the context manager, tensors are not fake."""
        with cpp_fake_tensor_mode():
            a = torch.randn(2, 3, device=device)
            self.assertTrue(torch._C._is_fake_tensor(a))
        b = torch.randn(2, 3, device=device)
        self.assertFalse(torch._C._is_fake_tensor(b))

    def test_mode_cleanup_on_exception(self, device):
        """Mode is cleaned up even when an exception occurs inside."""
        with self.assertRaises(RuntimeError):
            with cpp_fake_tensor_mode():
                raise RuntimeError("test error")
        b = torch.randn(2, 3, device=device)
        self.assertFalse(torch._C._is_fake_tensor(b))


instantiate_device_type_tests(TestFakeTensorCpp, globals(), only_for=("cpu", "cuda"))


if __name__ == "__main__":
    run_tests()
