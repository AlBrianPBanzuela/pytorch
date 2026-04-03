# Owner(s): ["module: dynamo"]
"""Tests for nb_bool / generic_bool: bool() via PyObject_IsTrue in Dynamo."""

import collections
import enum

import torch
import torch.nn
from torch.testing._internal.common_utils import make_dynamo_test, run_tests, TestCase


class NbBoolTests(TestCase):
    # --- Scalar constants (ConstantVariable path) ---

    @make_dynamo_test
    def test_int(self):
        self.assertFalse(bool(0))
        self.assertTrue(bool(1))
        self.assertTrue(bool(-1))

    @make_dynamo_test
    def test_float(self):
        self.assertFalse(bool(0.0))
        self.assertFalse(bool(-0.0))
        self.assertTrue(bool(3.14))

    @make_dynamo_test
    def test_none(self):
        self.assertFalse(bool(None))

    @make_dynamo_test
    def test_str(self):
        self.assertFalse(bool(""))
        self.assertTrue(bool("nonempty"))

    @make_dynamo_test
    def test_bytes(self):
        self.assertFalse(bool(b""))
        self.assertTrue(bool(b"hello"))

    @make_dynamo_test
    def test_bool(self):
        self.assertFalse(False)
        self.assertTrue(True)

    @make_dynamo_test
    def test_complex_zero(self):
        self.assertFalse(bool(0j))

    @make_dynamo_test
    def test_complex_nonzero(self):
        self.assertTrue(bool(1 + 2j))

    @make_dynamo_test
    def test_complex_real_nonzero_imag_zero(self):
        self.assertTrue(bool(1 + 0j))

    @make_dynamo_test
    def test_complex_real_zero_imag_nonzero(self):
        self.assertTrue(bool(0 + 1j))

    # --- Containers (length fallback / _bool_from_length path) ---

    @make_dynamo_test
    def test_empty_list(self):
        self.assertFalse(bool([]))

    @make_dynamo_test
    def test_nonempty_list(self):
        self.assertTrue(bool([1, 2, 3]))

    @make_dynamo_test
    def test_empty_dict(self):
        self.assertFalse(bool({}))

    @make_dynamo_test
    def test_nonempty_dict(self):
        self.assertTrue(bool({"a": 1}))

    @make_dynamo_test
    def test_empty_tuple(self):
        self.assertFalse(bool(()))

    @make_dynamo_test
    def test_nonempty_tuple(self):
        self.assertTrue(bool((1,)))

    @make_dynamo_test
    def test_empty_set(self):
        self.assertFalse(bool(set()))

    @make_dynamo_test
    def test_nonempty_set(self):
        self.assertTrue(bool({1, 2}))

    @make_dynamo_test
    def test_empty_frozenset(self):
        self.assertFalse(bool(frozenset()))

    @make_dynamo_test
    def test_nonempty_frozenset(self):
        self.assertTrue(bool(frozenset({1})))

    @make_dynamo_test
    def test_empty_range(self):
        self.assertFalse(bool(range(0)))

    @make_dynamo_test
    def test_nonempty_range(self):
        self.assertTrue(bool(range(5)))

    # --- dict subclasses ---

    @make_dynamo_test
    def test_empty_defaultdict(self):
        d = collections.defaultdict(int)
        self.assertFalse(bool(d))

    @make_dynamo_test
    def test_nonempty_defaultdict(self):
        d = collections.defaultdict(int, {"x": 1})
        self.assertTrue(bool(d))

    @make_dynamo_test
    def test_empty_counter(self):
        c = collections.Counter()
        self.assertFalse(bool(c))

    @make_dynamo_test
    def test_nonempty_counter(self):
        c = collections.Counter("abc")
        self.assertTrue(bool(c))

    # --- Enum (UserDefinedClassVariable / ConstantVariable path) ---

    @make_dynamo_test
    def test_enum_member(self):
        class Color(enum.Enum):
            RED = 1
            BLUE = 2

        self.assertTrue(bool(Color.RED))
        self.assertTrue(bool(Color.BLUE))

    # --- UserDefinedObjectVariable tests (torch.compile path) ---

    def test_user_defined_with_bool(self):
        class MyObj:
            def __init__(self, val):
                self.val = val

            def __bool__(self):
                return self.val > 0

        def fn(x, obj):
            return x + 1 if bool(obj) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x, MyObj(5)), compiled(x, MyObj(5)))
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x, MyObj(-1)), compiled(x, MyObj(-1)))

    def test_user_defined_with_len(self):
        class Container:
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

        def fn(x, c):
            return x + 1 if bool(c) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x, Container([1, 2])), compiled(x, Container([1, 2])))
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x, Container([])), compiled(x, Container([])))

    def test_user_defined_no_bool_no_len(self):
        class Plain:
            pass

        def fn(x, obj):
            return x + 1 if bool(obj) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, Plain()), compiled(x, Plain()))

    def test_user_defined_bool_returns_non_bool_raises(self):
        class BadBool:
            def __bool__(self):
                return 1  # noqa: PLE0305

        def fn(x, obj):
            return x + 1 if bool(obj) else x - 1

        with self.assertRaises(TypeError):
            bool(BadBool())
        with self.assertRaises(TypeError):
            torch.compile(fn, backend="eager")(torch.randn(4), BadBool())

    # --- Metaclass with __bool__ (UserDefinedClassVariable path) ---

    def test_metaclass_with_bool_false(self):
        class FalseMeta(type):
            def __bool__(cls):
                return False

        class A(metaclass=FalseMeta):
            pass

        def fn(x):
            return x + 1 if bool(A) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_metaclass_with_bool_default(self):
        class B:
            pass

        def fn(x):
            return x + 1 if bool(B) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    # --- nn.Module (NNModuleVariable path) ---

    def test_nn_module_nonempty(self):
        mod = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

        def fn(x):
            return x + 1 if bool(mod) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_nn_module_empty(self):
        mod = torch.nn.ModuleList()

        def fn(x):
            return x + 1 if bool(mod) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    # --- Tensor (TensorVariable path) ---

    def test_tensor_nonzero(self):
        def fn(x):
            t = torch.tensor(1)
            return x + 1 if bool(t) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_tensor_zero(self):
        def fn(x):
            t = torch.tensor(0)
            return x + 1 if bool(t) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))


if __name__ == "__main__":
    run_tests()
