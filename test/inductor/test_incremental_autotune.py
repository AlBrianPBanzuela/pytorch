# Owner(s): ["module: inductor"]

from unittest.mock import MagicMock

from torch._inductor.runtime.incremental._launcher import Launcher
from torch._inductor.test_case import run_tests, TestCase


def _make_launcher(name: str = "launcher") -> Launcher:
    fn = MagicMock()
    fn.return_value = f"result_{name}"
    return Launcher(fn=fn, config=f"config:{name}")


class LauncherTest(TestCase):
    def test_timing_empty(self):
        launcher = _make_launcher()
        self.assertEqual(launcher.timing, float("inf"))

    def test_sample_count(self):
        launcher = _make_launcher()
        launcher._add_timing(1.0)
        launcher._add_timing(2.0)
        self.assertEqual(launcher.sample_count, 2)

    def test_add_timing_sorted(self):
        launcher = _make_launcher()
        launcher._add_timing(3.0)
        launcher._add_timing(1.0)
        launcher._add_timing(2.0)
        self.assertEqual(launcher._timings, [1.0, 2.0, 3.0])

    def test_timing_median_odd(self):
        launcher = _make_launcher()
        for v in [5.0, 1.0, 3.0]:
            launcher._add_timing(v)
        self.assertAlmostEqual(launcher.timing, 3.0)

    def test_timing_median_even(self):
        launcher = _make_launcher()
        for v in [4.0, 2.0]:
            launcher._add_timing(v)
        self.assertAlmostEqual(launcher.timing, 3.0)

    def test_timing_median_ignores_outliers(self):
        launcher = _make_launcher()
        for v in [1.0, 2.0, 3.0, 4.0, 100.0]:
            launcher._add_timing(v)
        self.assertAlmostEqual(launcher.timing, 3.0)

    def test_dispatch_count_increments_on_call(self):
        launcher = _make_launcher()
        self.assertEqual(launcher.dispatch_count, 0)
        launcher(stream=0)
        self.assertEqual(launcher.dispatch_count, 1)
        launcher(stream=0)
        self.assertEqual(launcher.dispatch_count, 2)

    def test_call_delegates_to_fn(self):
        launcher = _make_launcher("test")
        result = launcher(1, 2, stream=0)
        self.assertEqual(result, "result_test")
        launcher._fn.assert_called_once_with(1, 2, stream=0)

    def test_metadata(self):
        launcher = Launcher(fn=lambda: None, key="value")
        self.assertEqual(launcher.metadata["key"], "value")


if __name__ == "__main__":
    run_tests()
