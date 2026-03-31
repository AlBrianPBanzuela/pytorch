# Owner(s): ["module: higher order operators"]
"""Tests for torch.utils.debug_log.debug_grad_log."""

import logging

import torch
from functorch.compile import aot_function, make_boxed_func
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.utils.debug_log import debug_grad_log


def nop(fx_g, _):
    return make_boxed_func(fx_g)


class _LogCapture(logging.Handler):
    """Logging handler that captures formatted log records."""

    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(self.format(record))


@skipIfTorchDynamo("debug_grad_log tests manage their own compilation")
class TestDebugGradLog(TestCase):
    def _add_log_capture(self):
        capture = _LogCapture()
        logger = logging.getLogger("torch.utils.debug_log")
        logger.addHandler(capture)
        logger.setLevel(logging.INFO)
        self.addCleanup(logger.removeHandler, capture)
        return capture

    def test_single_tensor_eager(self):
        """Backward gradient norm is logged for a single tensor."""
        capture = self._add_log_capture()

        x = torch.randn(3, requires_grad=True)
        y = x * 2
        debug_grad_log("single", y)
        y.sum().backward()

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 1)
        self.assertIn("[single][bwd]", bwd[0])
        self.assertIn("t0_grad_norm=", bwd[0])

    def test_multi_tensor_eager(self):
        """Backward gradient norms logged for multiple tensors, fires once."""
        capture = self._add_log_capture()

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        z = x * 2 + y * 3
        debug_grad_log("multi", x, y)
        z.sum().backward()

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 1)
        self.assertIn("[multi][bwd]", bwd[0])
        self.assertIn("t0_grad_norm=", bwd[0])
        self.assertIn("t1_grad_norm=", bwd[0])

    def test_multi_tensor_gradient_values(self):
        """Verify logged gradient norms match expected values."""
        capture = self._add_log_capture()

        # x grad = 2, y grad = 3; norms of scalar grads are themselves
        x = torch.tensor([2.0], requires_grad=True)
        y = torch.tensor([3.0], requires_grad=True)
        z = x * 2 + y * 3
        debug_grad_log("values", x, y)
        z.sum().backward()

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 1)
        # x_grad = 2.0, y_grad = 3.0
        self.assertIn("t0_grad_norm=2.0000", bwd[0])
        self.assertIn("t1_grad_norm=3.0000", bwd[0])

    def test_no_requires_grad_no_log(self):
        """No backward log when no tensor requires grad."""
        capture = self._add_log_capture()

        x = torch.randn(3, requires_grad=False)
        debug_grad_log("noop", x)

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 0)

    def test_aot_function_single_tensor(self):
        """Works under aot_function with a single tensor."""
        capture = self._add_log_capture()

        def f(x):
            y = x * 2
            debug_grad_log("aot", y)
            return y

        x = torch.randn(4, requires_grad=True)
        aot_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        out = aot_f(x)
        out.sum().backward()

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 1)
        self.assertIn("[aot][bwd]", bwd[0])

    def test_aot_function_multi_tensor(self):
        """Works under aot_function with multiple tensors."""
        capture = self._add_log_capture()

        def f(x, y):
            z = x * 2 + y * 3
            debug_grad_log("aot_multi", x, y)
            return z

        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=True)
        aot_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        out = aot_f(x, y)
        out.sum().backward()

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 1)
        self.assertIn("[aot_multi][bwd]", bwd[0])
        self.assertIn("t0_grad_norm=", bwd[0])
        self.assertIn("t1_grad_norm=", bwd[0])

    def test_compile_multi_tensor(self):
        """Works under torch.compile with multiple tensors."""
        capture = self._add_log_capture()
        torch._dynamo.reset()

        def f(x, y):
            z = x * 2 + y * 3
            debug_grad_log("compiled", x, y)
            return z

        compiled_f = torch.compile(f, backend="aot_eager", fullgraph=True)
        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=True)
        out = compiled_f(x, y)
        out.sum().backward()

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 1)
        self.assertIn("[compiled][bwd]", bwd[0])
        self.assertIn("t0_grad_norm=", bwd[0])
        self.assertIn("t1_grad_norm=", bwd[0])

    def test_forward_is_noop(self):
        """debug_grad_log does nothing in the forward pass."""
        capture = self._add_log_capture()

        x = torch.randn(3, requires_grad=True)
        debug_grad_log("fwd_check", x)

        # Before backward, no logs at all
        self.assertEqual(len(capture.records), 0)


if __name__ == "__main__":
    run_tests()
