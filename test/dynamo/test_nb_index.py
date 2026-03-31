# Owner(s): ["module: dynamo"]
"""Tests for nb_index_impl: unified __index__ / operator.index protocol in Dynamo."""

import operator

import torch
import torch._dynamo.testing
from torch.testing._internal.common_utils import make_dynamo_test, run_tests, TestCase


class NbIndexTests(TestCase):
    # --- int / bool (ConstantVariable) ---

    @make_dynamo_test
    def test_int_index(self):
        self.assertEqual(operator.index(5), 5)

    @make_dynamo_test
    def test_bool_index(self):
        self.assertEqual(operator.index(True), 1)

    @make_dynamo_test
    def test_int_dunder_index(self):
        self.assertEqual((5).__index__(), 5)

    # --- TypeError for non-index types ---

    def test_str_index_raises(self):
        def fn(x):
            try:
                return operator.index("hello")
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("cannot be interpreted as an integer", result)

    def test_float_index_raises(self):
        def fn(x):
            try:
                return operator.index(3.14)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("cannot be interpreted as an integer", result)

    def test_list_index_raises(self):
        def fn(x):
            try:
                return operator.index([1, 2])
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("cannot be interpreted as an integer", result)

    def test_none_index_raises(self):
        def fn(x):
            try:
                return operator.index(None)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("cannot be interpreted as an integer", result)

    # --- UserDefinedObjectVariable with __index__ ---

    def test_user_defined_index(self):
        class MyInt:
            def __init__(self, v):
                self.v = v

            def __index__(self):
                return self.v

        obj = MyInt(42)

        def fn(x):
            return operator.index(obj)

        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 42)

    def test_user_defined_dunder_index(self):
        class MyInt:
            def __init__(self, v):
                self.v = v

            def __index__(self):
                return self.v

        obj = MyInt(7)

        def fn(x):
            return obj.__index__()

        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 7)

    def test_user_defined_no_index_raises(self):
        class NoIndex:
            pass

        obj = NoIndex()

        def fn(x):
            try:
                return operator.index(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("cannot be interpreted as an integer", result)

    # --- nb_index used in list/tuple subscript (PyNumber_AsSsize_t) ---

    @make_dynamo_test
    def test_list_subscript_with_bool(self):
        lst = [10, 20, 30]
        self.assertEqual(lst[True], 20)

    @make_dynamo_test
    def test_tuple_subscript_with_bool(self):
        t = (10, 20, 30)
        self.assertEqual(t[True], 20)

    def test_list_subscript_with_user_defined_index(self):
        class MyIdx:
            def __index__(self):
                return 1

        idx = MyIdx()

        def fn(x):
            lst = [10, 20, 30]
            return lst[idx]

        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 20)

    def test_tuple_subscript_with_user_defined_index(self):
        class MyIdx:
            def __index__(self):
                return 2

        idx = MyIdx()

        def fn(x):
            t = (10, 20, 30)
            return t[idx]

        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 30)

    def test_list_subscript_with_no_index_raises(self):
        class NoIndex:
            pass

        obj = NoIndex()

        def fn(x):
            lst = [10, 20, 30]
            try:
                return lst[obj]
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("cannot be interpreted as an integer", result)


if __name__ == "__main__":
    run_tests()
