# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.testing
from torch._dynamo.dynamic_spec import IntSpec, IntSpecType
from torch.testing._internal.common_utils import run_tests, TestCase


class TestIntSpec(TestCase):
    # -- construction (direct) ---------------------------------------------

    def test_static_with_value(self):
        s = IntSpec("x", type=IntSpecType.STATIC, value=42)
        self.assertEqual(s.name, "x")
        self.assertEqual(s.type, IntSpecType.STATIC)
        self.assertEqual(s.value, 42)
        self.assertIsNone(s.min)
        self.assertIsNone(s.max)

    def test_static_without_value(self):
        s = IntSpec("x", type=IntSpecType.STATIC)
        self.assertEqual(s.value, None)

    def test_backed_with_bounds(self):
        s = IntSpec("batch", type=IntSpecType.BACKED, min=1, max=64)
        self.assertEqual(s.type, IntSpecType.BACKED)
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 64)
        self.assertIsNone(s.value)

    def test_backed_with_hint(self):
        s = IntSpec("b", type=IntSpecType.BACKED, backed_hint=32)
        self.assertEqual(s.backed_hint, 32)

    def test_unbacked_with_bounds(self):
        s = IntSpec("seq", type=IntSpecType.UNBACKED, min=1, max=2048)
        self.assertEqual(s.type, IntSpecType.UNBACKED)
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 2048)

    def test_unbacked_with_hint(self):
        s = IntSpec("seq", type=IntSpecType.UNBACKED, optimization_hint=512)
        self.assertEqual(s.optimization_hint, 512)

    def test_no_name(self):
        s = IntSpec(type=IntSpecType.STATIC, value=10)
        self.assertIsNone(s.name)

    def test_no_type_allowed(self):
        s = IntSpec("x")
        self.assertIsNone(s.type)

    # -- validation --------------------------------------------------------

    def test_static_rejects_min(self):
        with self.assertRaisesRegex(ValueError, "min/max.*STATIC"):
            IntSpec(type=IntSpecType.STATIC, min=1)

    def test_static_rejects_max(self):
        with self.assertRaisesRegex(ValueError, "min/max.*STATIC"):
            IntSpec(type=IntSpecType.STATIC, max=100)

    def test_static_rejects_optimization_hint(self):
        with self.assertRaisesRegex(ValueError, "optimization_hint.*UNBACKED"):
            IntSpec(type=IntSpecType.STATIC, optimization_hint=10)

    def test_static_rejects_backed_hint(self):
        with self.assertRaisesRegex(ValueError, "backed_hint.*BACKED"):
            IntSpec(type=IntSpecType.STATIC, backed_hint=10)

    def test_backed_rejects_value(self):
        with self.assertRaisesRegex(ValueError, "value.*STATIC"):
            IntSpec(type=IntSpecType.BACKED, value=42)

    def test_backed_rejects_optimization_hint(self):
        with self.assertRaisesRegex(ValueError, "optimization_hint.*UNBACKED"):
            IntSpec(type=IntSpecType.BACKED, optimization_hint=10)

    def test_unbacked_rejects_value(self):
        with self.assertRaisesRegex(ValueError, "value.*STATIC"):
            IntSpec(type=IntSpecType.UNBACKED, value=42)

    def test_unbacked_rejects_backed_hint(self):
        with self.assertRaisesRegex(ValueError, "backed_hint.*BACKED"):
            IntSpec(type=IntSpecType.UNBACKED, backed_hint=10)

    def test_backed_min_greater_than_max(self):
        with self.assertRaisesRegex(ValueError, "min must be <= max"):
            IntSpec(type=IntSpecType.BACKED, min=100, max=1)

    def test_unbacked_min_greater_than_max(self):
        with self.assertRaisesRegex(ValueError, "min must be <= max"):
            IntSpec(type=IntSpecType.UNBACKED, min=100, max=1)

    # -- fluent API --------------------------------------------------------

    def test_fluent_static(self):
        s = IntSpec("x").static(10)
        self.assertEqual(s.type, IntSpecType.STATIC)
        self.assertEqual(s.value, 10)

    def test_fluent_static_no_value(self):
        s = IntSpec().static()
        self.assertEqual(s.type, IntSpecType.STATIC)
        self.assertIsNone(s.value)

    def test_fluent_backed(self):
        s = IntSpec("batch").backed(min=1, max=64)
        self.assertEqual(s.type, IntSpecType.BACKED)
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 64)

    def test_fluent_backed_with_hint(self):
        s = IntSpec("b").backed(hint=32)
        self.assertEqual(s.backed_hint, 32)

    def test_fluent_unbacked(self):
        s = IntSpec("seq").unbacked(min=1, max=2048)
        self.assertEqual(s.type, IntSpecType.UNBACKED)
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 2048)

    def test_fluent_unbacked_with_hint(self):
        s = IntSpec("seq").unbacked(hint=512)
        self.assertEqual(s.optimization_hint, 512)

    def test_fluent_returns_self(self):
        s = IntSpec("x")
        result = s.backed(min=1)
        self.assertIs(result, s)

    def test_fluent_clears_previous_fields(self):
        s = IntSpec("x").backed(min=1, max=64)
        s.static(10)
        self.assertIsNone(s.min)
        self.assertIsNone(s.max)
        self.assertEqual(s.value, 10)

    def test_fluent_backed_rejects_bad_bounds(self):
        with self.assertRaisesRegex(ValueError, "min must be <= max"):
            IntSpec().backed(min=100, max=1)

    # -- match_assumptions -------------------------------------------------

    def test_match_static_exact(self):
        s = IntSpec().static(42)
        self.assertTrue(s.match_assumptions(42))
        self.assertFalse(s.match_assumptions(43))

    def test_match_static_no_value_errors(self):
        s = IntSpec().static()
        with self.assertRaisesRegex(ValueError, "no value set"):
            s.match_assumptions(42)

    def test_match_backed_in_bounds(self):
        s = IntSpec().backed(min=1, max=64)
        self.assertTrue(s.match_assumptions(1))
        self.assertTrue(s.match_assumptions(32))
        self.assertTrue(s.match_assumptions(64))

    def test_match_backed_out_of_bounds(self):
        s = IntSpec().backed(min=1, max=64)
        self.assertFalse(s.match_assumptions(0))
        self.assertFalse(s.match_assumptions(65))

    def test_match_backed_no_bounds(self):
        s = IntSpec().backed()
        self.assertTrue(s.match_assumptions(0))
        self.assertTrue(s.match_assumptions(999999))

    def test_match_unbacked_in_bounds(self):
        s = IntSpec().unbacked(min=10, max=100)
        self.assertTrue(s.match_assumptions(10))
        self.assertTrue(s.match_assumptions(50))
        self.assertTrue(s.match_assumptions(100))
        self.assertFalse(s.match_assumptions(9))
        self.assertFalse(s.match_assumptions(101))

    def test_match_no_type_errors(self):
        s = IntSpec()
        with self.assertRaisesRegex(ValueError, "type must be set"):
            s.match_assumptions(42)

    # -- repr / eq ---------------------------------------------------------

    def test_repr_static(self):
        r = repr(IntSpec("x", type=IntSpecType.STATIC, value=10))
        self.assertIn("name='x'", r)
        self.assertIn("type=static", r)
        self.assertIn("value=10", r)

    def test_repr_backed(self):
        r = repr(IntSpec("b", type=IntSpecType.BACKED, min=1, max=64))
        self.assertIn("type=backed", r)
        self.assertIn("min=1", r)
        self.assertIn("max=64", r)

    def test_eq(self):
        a = IntSpec("x", type=IntSpecType.BACKED, min=1, max=64)
        b = IntSpec("x", type=IntSpecType.BACKED, min=1, max=64)
        self.assertEqual(a, b)

    def test_neq_different_type(self):
        a = IntSpec("x", type=IntSpecType.BACKED)
        b = IntSpec("x", type=IntSpecType.STATIC)
        self.assertNotEqual(a, b)

    def test_neq_different_name(self):
        a = IntSpec("x", type=IntSpecType.BACKED)
        b = IntSpec("y", type=IntSpecType.BACKED)
        self.assertNotEqual(a, b)

    def test_eq_not_intspec(self):
        self.assertNotEqual(IntSpec().static(1), 1)

    def test_hashable(self):
        a = IntSpec("x", type=IntSpecType.BACKED, min=1)
        b = IntSpec("x", type=IntSpecType.BACKED, min=1)
        self.assertEqual(hash(a), hash(b))
        s = {a, b}
        self.assertEqual(len(s), 1)


class TestIntSpecCompile(TestCase):
    """End-to-end tests: IntSpec with torch.compile via mark_dynamic_spec."""

    def test_static_compile(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            return x + 1

        x = torch.randn(4, 3)
        torch._dynamo.decorators.mark_dynamic_spec(x, 0, IntSpec().static())
        result = fn(x)
        self.assertEqual(result, x + 1)
        self.assertEqual(cnt.frame_count, 1)

    def test_backed_compile(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            return x.sum(0)

        x1 = torch.randn(4, 3)
        torch._dynamo.decorators.mark_dynamic_spec(x1, 0, IntSpec("batch").backed())
        result1 = fn(x1)
        self.assertEqual(result1, x1.sum(0))

        # Different batch size should NOT trigger recompile
        x2 = torch.randn(8, 3)
        torch._dynamo.decorators.mark_dynamic_spec(x2, 0, IntSpec("batch").backed())
        result2 = fn(x2)
        self.assertEqual(result2, x2.sum(0))
        self.assertEqual(cnt.frame_count, 1)

    def test_unbacked_compile(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            return x.sum(0)

        x = torch.randn(4, 3)
        torch._dynamo.decorators.mark_dynamic_spec(x, 0, IntSpec("batch").unbacked())
        result = fn(x)
        self.assertEqual(result, x.sum(0))

    def test_no_type_errors(self):
        x = torch.randn(4, 3)
        with self.assertRaisesRegex(ValueError, "type must be set"):
            torch._dynamo.decorators.mark_dynamic_spec(x, 0, IntSpec())

    def test_not_intspec_errors(self):
        x = torch.randn(4, 3)
        with self.assertRaisesRegex(TypeError, "Expected IntSpec"):
            torch._dynamo.decorators.mark_dynamic_spec(x, 0, "bad")


if __name__ == "__main__":
    run_tests()
