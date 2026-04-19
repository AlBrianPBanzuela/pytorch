"""
Python runtime bridge for extern (non-Triton) CUDA kernels in cpp_wrapper mode.

Called from C++ via the CPython API.  The two entry points mirror the Triton
lazy-compile bridge in triton_lazy_compile.py:

  start_kernel_compile  – eval the kernel source at module-load time
  run_kernel            – invoke the compiled kernel on every launch
"""

from __future__ import annotations

import logging
from typing import Any


log = logging.getLogger(__name__)


def _get_async_compile():
    from torch._inductor.async_compile import AsyncCompile

    return AsyncCompile()


class _SynchronousExternCompile:
    def cutedsl(self, kernel_name: str, source_code: str) -> Any:
        import torch._inductor.codecache
        from torch._inductor.codegen.cutedsl.cutedsl_kernel import (
            CuteDSLKernelWrapper,
            MAIN_SUFFIX,
        )

        key, path = torch._inductor.codecache.PyCodeCache.write(source_code)
        mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)
        main_func_name = f"{kernel_name}_{MAIN_SUFFIX}"
        if not hasattr(mod, main_func_name):
            available = [name for name in dir(mod) if callable(getattr(mod, name))]
            raise RuntimeError(
                f"Could not find CuteDSL main kernel function '{main_func_name}'. Available callables: {available}"
            )

        return CuteDSLKernelWrapper(
            getattr(mod, main_func_name),
            kernel_path=path,
            module=mod,
            export_jit_name=getattr(mod, "__inductor_export_jit_name__", None),
        )


def _preload_cutedsl_modules() -> None:
    import cuda.bindings.driver  # noqa: F401  # pyrefly: ignore [missing-import]
    import cutlass  # noqa: F401
    import cutlass.cute  # noqa: F401
    from cutlass._mlir.dialects import math as _mlir_math  # noqa: F401

    from torch._inductor.codegen.cutedsl import _cutedsl_utils  # noqa: F401


def start_kernel_compile(
    pending_kernels: dict[str, Any], kernel_name: str, kernel_source_path: str
) -> None:
    """Start compilation of an extern kernel.

    Called from C++ at module initialisation time for each extern kernel.
    Reads the sidecar source file (whose content is an ``async_compile.cutedsl(...)``
    expression), evaluates it to obtain a kernel wrapper or future, and stores
    the result in *pending_kernels* for later retrieval by ``run_kernel``.
    """
    with open(kernel_source_path) as f:
        kernel_source = f.read()
    if "import cutlass" in kernel_source and "cutlass.cute" in kernel_source:
        _preload_cutedsl_modules()
        async_compile = _SynchronousExternCompile()  # noqa: F841
    else:
        async_compile = _get_async_compile()  # noqa: F841 (used by eval below)
    kernel_obj = eval(kernel_source.strip())  # noqa: S307

    pending_kernels[kernel_name] = {
        "kernel_obj": kernel_obj,
        "cabi_artifact": None,
    }


def _resolve_kernel_entry(
    pending_kernels: dict[str, Any], kernel_name: str
) -> dict[str, Any]:
    entry = pending_kernels[kernel_name]
    if not isinstance(entry, dict):
        entry = {"kernel_obj": entry, "cabi_artifact": None}
        pending_kernels[kernel_name] = entry

    kernel_obj = entry["kernel_obj"]
    if hasattr(kernel_obj, "result"):
        kernel_obj = kernel_obj.result()
        entry["kernel_obj"] = kernel_obj
    return entry


def prepare_cabi_kernel(
    pending_kernels: dict[str, Any],
    kernel_name: str,
    stream: int,
    args: list[Any],
) -> tuple[str, str]:
    entry = _resolve_kernel_entry(pending_kernels, kernel_name)

    if entry["cabi_artifact"] is None:
        entry["cabi_artifact"] = entry["kernel_obj"].prepare_cabi_kernel(
            kernel_name, args, stream
        )

    artifact = entry["cabi_artifact"]
    return artifact["shared_object_path"], artifact["symbol_name"]


def run_kernel(
    pending_kernels: dict[str, Any],
    kernel_name: str,
    stream: int,
    args: list[Any],
) -> None:
    """Invoke a compiled extern kernel.

    Called from C++ on every kernel launch.  On the first call the kernel
    object may still be a ``LambdaFuture``; if so it is resolved and the dict
    entry is replaced with the concrete wrapper so that subsequent calls skip
    the future resolution.
    """
    entry = _resolve_kernel_entry(pending_kernels, kernel_name)
    entry["kernel_obj"].run(*args, stream=stream)
