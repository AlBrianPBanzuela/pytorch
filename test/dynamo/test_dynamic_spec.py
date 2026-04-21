# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.testing
from torch._dynamo.dynamic_spec import IntSpec, IntSpecType, TensorSpec
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestIntSpecConstruction(TestCase):
    """Construction via the classmethod factories."""

    def test_static(self):
        s = IntSpec.static("x", value=10)
        self.assertEqual(s.name, "x")
        self.assertEqual(s.type, IntSpecType.STATIC)
        self.assertEqual(s.value, 10)

    def test_static_no_value(self):
        s = IntSpec.static()
        self.assertEqual(s.type, IntSpecType.STATIC)
        self.assertIsNone(s.value)

    def test_backed(self):
        s = IntSpec.backed("batch", min=1, max=64, guarding_hint=32)
        self.assertEqual(s.name, "batch")
        self.assertEqual(s.type, IntSpecType.BACKED)
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 64)
        self.assertEqual(s.guarding_hint, 32)

    def test_unbacked(self):
        s = IntSpec.unbacked("seq", min=1, max=2048, optimization_hint=512)
        self.assertEqual(s.type, IntSpecType.UNBACKED)
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 2048)
        self.assertEqual(s.optimization_hint, 512)

    def test_type_required_on_init(self):
        with self.assertRaises(TypeError):
            IntSpec("x")  # no type kwarg

    def test_type_not_none(self):
        with self.assertRaises(TypeError):
            IntSpec("x", type=None)  # type: ignore[arg-type]


class TestIntSpecImmutable(TestCase):
    """Once constructed, the mode is fixed; per-mode accessors enforce it."""

    def test_value_only_on_static(self):
        s = IntSpec.backed("x")
        with self.assertRaisesRegex(AttributeError, "STATIC"):
            _ = s.value

    def test_guarding_hint_only_on_backed(self):
        s = IntSpec.unbacked("x")
        with self.assertRaisesRegex(AttributeError, "BACKED"):
            _ = s.guarding_hint
        s2 = IntSpec.static("x")
        with self.assertRaisesRegex(AttributeError, "BACKED"):
            _ = s2.guarding_hint

    def test_optimization_hint_only_on_unbacked(self):
        s = IntSpec.backed("x")
        with self.assertRaisesRegex(AttributeError, "UNBACKED"):
            _ = s.optimization_hint
        s2 = IntSpec.static("x")
        with self.assertRaisesRegex(AttributeError, "UNBACKED"):
            _ = s2.optimization_hint


class TestIntSpecValidation(TestCase):
    """Cross-parameter validation rejects bad combinations."""

    def test_static_rejects_min(self):
        with self.assertRaisesRegex(ValueError, "min/max.*STATIC"):
            IntSpec("x", type=IntSpecType.STATIC, min=1)

    def test_static_rejects_max(self):
        with self.assertRaisesRegex(ValueError, "min/max.*STATIC"):
            IntSpec("x", type=IntSpecType.STATIC, max=100)

    def test_static_rejects_guarding_hint(self):
        with self.assertRaisesRegex(ValueError, "guarding_hint.*BACKED"):
            IntSpec("x", type=IntSpecType.STATIC, guarding_hint=10)

    def test_static_rejects_optimization_hint(self):
        with self.assertRaisesRegex(ValueError, "optimization_hint.*UNBACKED"):
            IntSpec("x", type=IntSpecType.STATIC, optimization_hint=10)

    def test_backed_rejects_value(self):
        with self.assertRaisesRegex(ValueError, "value.*STATIC"):
            IntSpec("x", type=IntSpecType.BACKED, value=42)

    def test_backed_rejects_optimization_hint(self):
        with self.assertRaisesRegex(ValueError, "optimization_hint.*UNBACKED"):
            IntSpec("x", type=IntSpecType.BACKED, optimization_hint=10)

    def test_unbacked_rejects_value(self):
        with self.assertRaisesRegex(ValueError, "value.*STATIC"):
            IntSpec("x", type=IntSpecType.UNBACKED, value=42)

    def test_unbacked_rejects_guarding_hint(self):
        with self.assertRaisesRegex(ValueError, "guarding_hint.*BACKED"):
            IntSpec("x", type=IntSpecType.UNBACKED, guarding_hint=10)

    def test_backed_min_greater_than_max(self):
        with self.assertRaisesRegex(ValueError, "min must be <= max"):
            IntSpec.backed("x", min=100, max=1)

    def test_unbacked_min_greater_than_max(self):
        with self.assertRaisesRegex(ValueError, "min must be <= max"):
            IntSpec.unbacked("x", min=100, max=1)


class TestIntSpecEq(TestCase):
    """__eq__ and __hash__."""

    def test_eq(self):
        a = IntSpec.backed("x", min=1, max=64)
        b = IntSpec.backed("x", min=1, max=64)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_neq_different_type(self):
        self.assertNotEqual(IntSpec.backed("x"), IntSpec.static("x"))

    def test_neq_different_name(self):
        self.assertNotEqual(IntSpec.backed("x"), IntSpec.backed("y"))

    def test_eq_not_intspec(self):
        self.assertNotEqual(IntSpec.static("x", value=1), 1)


class TestTensorSpecConstruction(TestCase):
    """Construction and list-like interface."""

    def test_basic(self):
        ts = TensorSpec(3)
        self.assertEqual(ts.rank, 3)
        self.assertEqual(len(ts), 3)
        for spec in ts:
            self.assertIsNone(spec)

    def test_zero_rank(self):
        ts = TensorSpec(0)
        self.assertEqual(ts.rank, 0)
        self.assertEqual(len(ts), 0)

    def test_negative_rank(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            TensorSpec(-1)

    def test_from_list(self):
        specs = [IntSpec.static(value=10), None, IntSpec.backed(min=1)]
        ts = TensorSpec.from_list(specs)
        self.assertEqual(ts.rank, 3)
        self.assertEqual(ts[0], IntSpec.static(value=10))
        self.assertIsNone(ts[1])

    def test_getitem_setitem(self):
        ts = TensorSpec(2)
        spec = IntSpec.backed("batch", min=1)
        ts[0] = spec
        self.assertEqual(ts[0], spec)
        self.assertIsNone(ts[1])

    def test_set_fluent(self):
        ts = TensorSpec(3)
        result = ts.set(0, IntSpec.static(value=10))
        self.assertIs(result, ts)
        self.assertEqual(ts[0], IntSpec.static(value=10))

    def test_iter(self):
        ts = TensorSpec(2)
        ts[0] = IntSpec.static(value=5)
        items = list(ts)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0], IntSpec.static(value=5))
        self.assertIsNone(items[1])

    def test_index_out_of_range(self):
        ts = TensorSpec(2)
        with self.assertRaises(IndexError):
            ts[5]

    def test_sparse_set(self):
        ts = TensorSpec(4)
        ts.set(1, IntSpec.backed("h"))
        ts.set(3, IntSpec.backed("w"))
        self.assertIsNone(ts[0])
        self.assertIsNotNone(ts[1])
        self.assertIsNone(ts[2])
        self.assertIsNotNone(ts[3])


class TestTensorSpecEq(TestCase):
    """__eq__ and __hash__."""

    def test_eq(self):
        a = TensorSpec(2).set(0, IntSpec.static(value=10))
        b = TensorSpec(2).set(0, IntSpec.static(value=10))
        self.assertEqual(a, b)

    def test_neq_different_rank(self):
        self.assertNotEqual(TensorSpec(2), TensorSpec(3))

    def test_neq_different_specs(self):
        a = TensorSpec(2).set(0, IntSpec.static(value=10))
        b = TensorSpec(2).set(0, IntSpec.static(value=20))
        self.assertNotEqual(a, b)


class TestTensorSpecCompile(TestCase):
    """torch.compile(dynamic_shapes=...) with TensorSpec."""

    def test_tensorspec_backed_dim(self):
        torch._dynamo.reset()
        ts = TensorSpec(2).set(0, IntSpec.backed("batch"))
        fn = torch.compile(
            lambda x: x.sum(0),
            backend="eager",
            dynamic_shapes={"x": ts},
        )
        for n in [4, 8, 16]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x.sum(0))

    def test_tensorspec_mixed_dims(self):
        torch._dynamo.reset()
        ts = TensorSpec(2).set(0, IntSpec.backed("batch")).set(1, IntSpec.static())
        fn = torch.compile(
            lambda x: x + 1,
            backend="eager",
            dynamic_shapes={"x": ts},
        )
        for n in [4, 8, 16]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x + 1)

    def test_tensorspec_partial_spec(self):
        torch._dynamo.reset()
        ts = TensorSpec(2).set(0, IntSpec.backed("batch"))
        fn = torch.compile(
            lambda x: x.sum(0),
            backend="eager",
            dynamic_shapes={"x": ts},
        )
        for n in [4, 8]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x.sum(0))

    @skipIfTorchDynamo("frame_count unreliable when dynamo traces the test")
    def test_tensorspec_limits_recompile(self):
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()
        ts = TensorSpec(2).set(0, IntSpec.backed("batch"))
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=cnt,
            dynamic_shapes={"x": ts},
        )
        for n in [4, 8, 16, 32]:
            fn(torch.randn(n, 3))
        # BACKED currently lowers to maybe_mark_dynamic: first call specializes
        # static, second promotes to dynamic, then cached.
        self.assertLessEqual(cnt.frame_count, 2)


class TestIntSpecCompile(TestCase):
    """torch.compile(dynamic_shapes=...) with IntSpec — functional tests.

    TODO(follow-up integration PR): replace recompile-count checks with
    graph-inspection (backed-symbol presence, DDE on unbacked branching),
    and cover precedence vs compile(dynamic=True/False).
    """

    @skipIfTorchDynamo("frame_count unreliable when dynamo traces the test")
    def test_static_recompiles_per_shape(self):
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(
            lambda x: x + 1,
            backend=cnt,
            dynamic_shapes={"x": {0: IntSpec.static()}},
        )
        fn(torch.randn(4, 3))
        fn(torch.randn(8, 3))
        fn(torch.randn(4, 3))  # cache hit
        self.assertEqual(cnt.frame_count, 2)

    @skipIfTorchDynamo("frame_count unreliable when dynamo traces the test")
    def test_backed_limits_recompile(self):
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=cnt,
            dynamic_shapes={"x": {0: IntSpec.backed("batch")}},
        )
        for n in [4, 8, 16, 32, 64]:
            fn(torch.randn(n, 3))
        # BACKED currently lowers to maybe_mark_dynamic: first call specializes
        # static, second promotes to dynamic, then cached.
        self.assertLessEqual(cnt.frame_count, 2)

    @skipIfTorchDynamo("frame_count unreliable when dynamo traces the test")
    def test_unbacked_no_recompile(self):
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=cnt,
            dynamic_shapes={"x": {0: IntSpec.unbacked("batch")}},
        )
        for n in [4, 8, 16, 32]:
            fn(torch.randn(n, 3))
        self.assertEqual(cnt.frame_count, 1)

    def test_list_form(self):
        torch._dynamo.reset()
        fn = torch.compile(
            lambda x: x + 1,
            backend="eager",
            dynamic_shapes={"x": [IntSpec.backed("batch"), IntSpec.static()]},
        )
        for n in [4, 8, 16]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x + 1)

    def test_none_entry_inherits_context(self):
        """A None entry in a list-form spec should not mark the dim."""
        torch._dynamo.reset()
        fn = torch.compile(
            lambda x: x + 1,
            backend="eager",
            dynamic_shapes={"x": [IntSpec.backed("batch"), None]},
        )
        x = torch.randn(4, 3)
        self.assertEqual(fn(x), x + 1)


if __name__ == "__main__":
    run_tests()
