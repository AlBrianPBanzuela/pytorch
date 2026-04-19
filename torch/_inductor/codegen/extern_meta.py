"""Structured metadata for extern (non-Triton) CUDA kernels in cpp_wrapper."""

from __future__ import annotations

import dataclasses
import enum


class ExternKernelBackend(enum.Enum):
    CUTEDSL = "cutedsl"
    CUTLASS = "cutlass"
    OTHER = "other"


class ExternKernelLaunch(enum.Enum):
    PYTHON = "python"
    CUBIN = "cubin"
    C_ABI = "c_abi"


@dataclasses.dataclass(frozen=True)
class ExternMeta:
    backend: ExternKernelBackend
    arg_names: list[str]
    launch: ExternKernelLaunch = ExternKernelLaunch.PYTHON
    export_jit_name: str | None = None
