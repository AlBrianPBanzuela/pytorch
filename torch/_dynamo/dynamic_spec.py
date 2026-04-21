"""Dynamic shape specification types for ``torch.compile`` and ``torch.export``.

Provides :class:`IntSpec` for fine-grained control over whether an integer
(dimension size or scalar argument) is treated as static, backed, or unbacked
during compilation.

Backed vs. unbacked
-------------------
``torch.compile`` provides two kinds of dynamic shapes: ``backed`` and
``unbacked``. ``torch.compile`` guards on ``backed`` dynamic shapes and does
not provide a guarantee that no guards will be added to them. User code,
dynamo, inductor, and autograd all can add guards when tracing through
branching, e.g. ``if x.size() > 10``. Moreover, for 0/1 specializations,
backed symbols are specialized unconditionally to ``0``, ``1``, or ``>=2``
even without encountering a branching on those ranges.

On the contrary, ``unbacked`` dynamic shapes are guaranteed not to be guarded
on and are not 0/1 specialized. However, there is a possibility of throwing a
data-dependent error when a branch that requires their value is encountered
and no explicit unbacked handling is defined. The framework is converging to
a state where it won't throw DDE but rather pick general paths. One downside
of using unbacked is missed optimization opportunities due to either perf
bugs or picking general paths, or using a fixed non-example input-based hint.
An example of picking general paths is assuming input not contiguous in
functions called ``contiguous()`` and ``reshape()`` when it cannot be
symbolically proven, with a change of introducing a clone.

For more info see
https://dev-discuss.pytorch.org/t/backed-to-unbacked-from-guardable-to-guardless-shapes-in-pytorch/3333.

.. TODO::

    Expand this documentation once ``TensorSpec`` and ``ModelSpec`` land, with
    end-to-end examples covering per-tensor and per-model specifications.
"""

import enum
from collections.abc import Iterator
from typing import Any


__all__ = ["IntSpecType", "IntSpec", "TensorSpec"]


class IntSpecType(enum.Enum):
    """How an integer should be treated during compilation.

    STATIC
        Treat as a compile-time constant. Recompiles if the value changes.
    BACKED
        Symbolic with a backing hint. Guards and 0/1 specialization are
        permitted; user code or the compiler may install constraints.
    UNBACKED
        Symbolic with no backing value. Guaranteed not to be guarded on and
        not 0/1 specialized; branching on the value may raise a
        data-dependent error.
    """

    STATIC = "static"
    BACKED = "backed"
    UNBACKED = "unbacked"


class IntSpec:
    """Shape specification for a single integer (dimension size or scalar arg).

    Constructed via one of the three mode-specific classmethod factories —
    :meth:`static`, :meth:`backed`, :meth:`unbacked`. The mode is required at
    construction time and cannot be changed afterwards.

    Example::

        IntSpec.static("x", value=10)
        IntSpec.backed("batch", min=1, max=64, guarding_hint=32)
        IntSpec.unbacked("seq", min=1, max=2048, optimization_hint=512)

    The ``__init__`` constructor is used by the factories and by callers that
    need to round-trip a spec via its fields; external users should prefer
    the factories.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        type: IntSpecType,
        min: int | None = None,
        max: int | None = None,
        value: int | None = None,
        guarding_hint: int | None = None,
        optimization_hint: int | None = None,
    ) -> None:
        if not isinstance(type, IntSpecType):
            raise TypeError(f"IntSpec.type must be an IntSpecType, got {type!r}")
        self.name = name
        self._type = type
        self._min = min
        self._max = max
        self._value = value
        self._guarding_hint = guarding_hint
        self._optimization_hint = optimization_hint
        self._validate()

    # -- validation --------------------------------------------------------

    def _validate(self) -> None:
        if self._type is IntSpecType.STATIC:
            if self._min is not None or self._max is not None:
                raise ValueError(
                    "min/max are only valid for BACKED/UNBACKED IntSpec, not STATIC"
                )
            if self._guarding_hint is not None:
                raise ValueError("guarding_hint is only valid for BACKED IntSpec")
            if self._optimization_hint is not None:
                raise ValueError("optimization_hint is only valid for UNBACKED IntSpec")
        elif self._type is IntSpecType.BACKED:
            if self._value is not None:
                raise ValueError("value is only valid for STATIC IntSpec")
            if self._optimization_hint is not None:
                raise ValueError("optimization_hint is only valid for UNBACKED IntSpec")
            if (
                self._min is not None
                and self._max is not None
                and self._min > self._max
            ):
                raise ValueError(
                    f"min must be <= max, got min={self._min}, max={self._max}"
                )
        elif self._type is IntSpecType.UNBACKED:
            if self._value is not None:
                raise ValueError("value is only valid for STATIC IntSpec")
            if self._guarding_hint is not None:
                raise ValueError("guarding_hint is only valid for BACKED IntSpec")
            if (
                self._min is not None
                and self._max is not None
                and self._min > self._max
            ):
                raise ValueError(
                    f"min must be <= max, got min={self._min}, max={self._max}"
                )

    # -- factories ---------------------------------------------------------

    @classmethod
    def static(cls, name: str | None = None, *, value: int | None = None) -> "IntSpec":
        """Construct a STATIC :class:`IntSpec`.

        ``value`` pins a concrete size; if ``None`` the value is taken from
        the example input at compile time.
        """
        return cls(name, type=IntSpecType.STATIC, value=value)

    @classmethod
    def backed(
        cls,
        name: str | None = None,
        *,
        min: int | None = None,
        max: int | None = None,
        guarding_hint: int | None = None,
    ) -> "IntSpec":
        """Construct a BACKED :class:`IntSpec`.

        ``guarding_hint`` is the concrete value the symbolic shape
        environment substitutes when a hint is needed for reasoning or
        codegen.
        """
        return cls(
            name,
            type=IntSpecType.BACKED,
            min=min,
            max=max,
            guarding_hint=guarding_hint,
        )

    @classmethod
    def unbacked(
        cls,
        name: str | None = None,
        *,
        min: int | None = None,
        max: int | None = None,
        optimization_hint: int | None = None,
    ) -> "IntSpec":
        """Construct an UNBACKED :class:`IntSpec`.

        ``optimization_hint`` is used by downstream codegen (e.g. inductor
        autotuning) only; it never participates in symbolic reasoning.
        """
        return cls(
            name,
            type=IntSpecType.UNBACKED,
            min=min,
            max=max,
            optimization_hint=optimization_hint,
        )

    # -- read-only properties ----------------------------------------------

    @property
    def type(self) -> IntSpecType:
        return self._type

    @property
    def min(self) -> int | None:
        return self._min

    @property
    def max(self) -> int | None:
        return self._max

    @property
    def value(self) -> int | None:
        if self._type is not IntSpecType.STATIC:
            raise AttributeError(
                f"value is only defined for STATIC IntSpec, got {self._type.value}"
            )
        return self._value

    @property
    def guarding_hint(self) -> int | None:
        if self._type is not IntSpecType.BACKED:
            raise AttributeError(
                f"guarding_hint is only defined for BACKED IntSpec, got {self._type.value}"
            )
        return self._guarding_hint

    @property
    def optimization_hint(self) -> int | None:
        if self._type is not IntSpecType.UNBACKED:
            raise AttributeError(
                f"optimization_hint is only defined for UNBACKED IntSpec, got {self._type.value}"
            )
        return self._optimization_hint

    # -- dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        parts: list[str] = []
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        parts.append(f"type={self._type.value}")
        if self._value is not None:
            parts.append(f"value={self._value}")
        if self._min is not None:
            parts.append(f"min={self._min}")
        if self._max is not None:
            parts.append(f"max={self._max}")
        if self._guarding_hint is not None:
            parts.append(f"guarding_hint={self._guarding_hint}")
        if self._optimization_hint is not None:
            parts.append(f"optimization_hint={self._optimization_hint}")
        return f"IntSpec({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntSpec):
            return NotImplemented
        return (
            self.name == other.name
            and self._type == other._type
            and self._min == other._min
            and self._max == other._max
            and self._value == other._value
            and self._guarding_hint == other._guarding_hint
            and self._optimization_hint == other._optimization_hint
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self._type,
                self._min,
                self._max,
                self._value,
                self._guarding_hint,
                self._optimization_hint,
            )
        )


class TensorSpec:
    """Per-dimension shape specification for a tensor.

    A list-like container of ``IntSpec | None`` with length equal to the
    tensor's rank. ``None`` entries inherit the default dynamism policy from
    the compile context.

    Example::

        ts = TensorSpec(3)
        ts.set(0, IntSpec.backed("batch", min=1, max=64))
        # dims 1 and 2 are None -> inherit context default
    """

    def __init__(self, rank: int) -> None:
        if rank < 0:
            raise ValueError(f"rank must be non-negative, got {rank}")
        self._rank = rank
        self._specs: list[IntSpec | None] = [None] * rank

    @classmethod
    def from_list(cls, specs: list[IntSpec | None]) -> "TensorSpec":
        """Construct from an existing list of specs."""
        ts = cls(len(specs))
        ts._specs = list(specs)
        return ts

    @property
    def rank(self) -> int:
        return self._rank

    def set(self, index: int, spec: IntSpec) -> "TensorSpec":
        """Set the spec at ``index`` and return ``self`` for chaining."""
        self._specs[index] = spec
        return self

    def __getitem__(self, index: int) -> IntSpec | None:
        return self._specs[index]

    def __setitem__(self, index: int, spec: IntSpec | None) -> None:
        self._specs[index] = spec

    def __len__(self) -> int:
        return self._rank

    def __iter__(self) -> Iterator[IntSpec | None]:
        return iter(self._specs)

    def __repr__(self) -> str:
        specified = [
            f"{i}: {spec!r}" for i, spec in enumerate(self._specs) if spec is not None
        ]
        return f"TensorSpec(rank={self._rank}, {{{', '.join(specified)}}})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorSpec):
            return NotImplemented
        return self._rank == other._rank and self._specs == other._specs

    def __hash__(self) -> int:
        return hash((self._rank, tuple(self._specs)))


# TODO: temporary scaffolding. laithsakka flagged (PR review) that translating
# specs into tensor properties via mark_*:
#   - does not work for scalar int inputs (a primary IntSpec use case),
#   - silently installs guards that haven't been decided on.
# Replace with proper plumbing through the compile context in the follow-up
# integration PR, then delete _apply_intspec_to_tensor and
# _apply_dynamic_shapes.
def _apply_intspec_to_tensor(tensor: Any, shape_spec: Any) -> None:
    """Apply per-dimension IntSpec entries to a tensor via ``mark_*``."""
    from torch._dynamo.decorators import mark_static, mark_unbacked, maybe_mark_dynamic

    if isinstance(shape_spec, TensorSpec):
        items: Any = enumerate(shape_spec)
    elif isinstance(shape_spec, dict):
        items = shape_spec.items()
    elif isinstance(shape_spec, (list, tuple)):
        items = enumerate(shape_spec)
    else:
        return

    for idx, spec in items:
        if spec is None:
            continue
        if not isinstance(spec, IntSpec):
            raise TypeError(
                f"Expected IntSpec or None in dynamic_shapes, got {type(spec).__name__}"
            )
        if spec.type is IntSpecType.STATIC:
            mark_static(tensor, idx)
        elif spec.type is IntSpecType.BACKED:
            maybe_mark_dynamic(tensor, idx)
        elif spec.type is IntSpecType.UNBACKED:
            mark_unbacked(tensor, idx)


def _apply_dynamic_shapes(
    compiled: Any, original: Any, dynamic_shapes: dict[str, Any]
) -> Any:
    """Wrap a compiled callable to apply ``dynamic_shapes`` IntSpec on each call.

    The wrapper is decorated with :func:`torch._dynamo.disable` so that dynamo
    does not attempt to trace the tensor-marking logic. The inner
    ``compiled()`` call re-enters dynamo normally.
    """
    import functools
    import inspect

    import torch
    import torch._dynamo

    sig = inspect.signature(
        original.forward if isinstance(original, torch.nn.Module) else original
    )

    @torch._dynamo.disable
    @functools.wraps(
        compiled if not isinstance(compiled, torch.nn.Module) else compiled.forward
    )
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, shape_spec in dynamic_shapes.items():
            if name in bound.arguments:
                arg = bound.arguments[name]
                if isinstance(arg, torch.Tensor):
                    _apply_intspec_to_tensor(arg, shape_spec)
        return compiled(*bound.args, **bound.kwargs)

    if isinstance(compiled, torch.nn.Module):
        compiled.forward = wrapper  # type: ignore[method-assign]
        return compiled
    return wrapper
