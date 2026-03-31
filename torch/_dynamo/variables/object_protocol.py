"""
Dynamo implementations of CPython's PyObject_* default slot algorithms.

Analogous to CPython's Objects/object.c, this module holds the general
comparison dispatch machinery that is independent of any specific type.
Per-type richcompare_impl hooks live in their respective VT files.
"""

from typing import TYPE_CHECKING

from ..utils import istype
from .base import NO_SUCH_SUBOBJ, raise_type_error_exc, VariableTracker
from .constant import CONSTANT_VARIABLE_FALSE, CONSTANT_VARIABLE_TRUE


type_error = raise_type_error_exc


if TYPE_CHECKING:
    from ..symbolic_convert import InstructionTranslator


def vt_identity_compare(
    left: VariableTracker,
    right: VariableTracker,
) -> "VariableTracker | None":
    """Try to determine Python identity (left is right) at trace time.

    Returns ConstantVariable(True/False) if determinable, else None.
    Mirrors the logic in BuiltinVariable's handle_is handler.
    """
    if left is right:
        return CONSTANT_VARIABLE_TRUE

    left_val = left.get_real_python_backed_value()
    right_val = right.get_real_python_backed_value()
    left_known = left_val is not NO_SUCH_SUBOBJ
    right_known = right_val is not NO_SUCH_SUBOBJ

    if left_known and right_known:
        return (
            CONSTANT_VARIABLE_TRUE if left_val is right_val else CONSTANT_VARIABLE_FALSE
        )

    # One side has a concrete backing object, the other doesn't — they can't
    # be the same object.
    if left_known != right_known:
        return CONSTANT_VARIABLE_FALSE

    # Mutable containers created during tracing: VT identity = Python identity.
    from .dicts import ConstDictVariable
    from .lists import ListVariable

    if isinstance(left, (ConstDictVariable, ListVariable)):
        return CONSTANT_VARIABLE_FALSE

    # Different Python types can never be the same object.
    try:
        if left.python_type() is not right.python_type():
            return CONSTANT_VARIABLE_FALSE
    except NotImplementedError:
        pass

    # Different exception types are never identical.
    from .. import variables

    if (
        istype(left, variables.ExceptionVariable)
        and istype(right, variables.ExceptionVariable)
        and left.exc_type is not right.exc_type  # type: ignore[attr-defined]
    ):
        return CONSTANT_VARIABLE_FALSE

    return None


def vt_implements_slot(
    obj: "VariableTracker",
    dunder: str,
    impl_method: str,
) -> bool:
    """
    Check whether obj implements a CPython slot, identified by both its Python
    dunder name and the corresponding VT impl method name.

    - UserDefinedObjectVariable: check whether the underlying class defines dunder.
    - ConstantVariable: check hasattr on the wrapped value.
    - All other VTs: check whether the subclass overrides impl_method.
    """
    from .base import VariableTracker
    from .constant import ConstantVariable
    from .user_defined import UserDefinedObjectVariable

    if istype(obj, UserDefinedObjectVariable):
        return obj._maybe_get_baseclass_method(dunder) is not None
    elif istype(obj, ConstantVariable):
        return hasattr(obj.value, dunder)
    else:
        m1 = getattr(obj.__class__, impl_method)
        m2 = getattr(VariableTracker, impl_method)
        return m1 is not m2


def vt_implements_sq_length(obj: "VariableTracker") -> bool:
    return vt_implements_slot(obj, "__len__", "sq_length")


def vt_implements_mp_length(obj: "VariableTracker") -> bool:
    return vt_implements_slot(obj, "__len__", "mp_length")


def vt_mapping_size(
    tx: "InstructionTranslator", obj: "VariableTracker"
) -> "VariableTracker":
    # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/abstract.c#L2308-L2330
    if vt_implements_mp_length(obj):
        return obj.mp_length(tx)

    if vt_implements_sq_length(obj):
        type_error(tx, f"{obj.python_type_name()} is not a mapping")

    type_error(tx, f"object of type {obj.python_type_name()} has no len()")


def generic_len(
    tx: "InstructionTranslator", obj: "VariableTracker"
) -> "VariableTracker":
    # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/abstract.c#L53-L69
    """
    Implements PyObject_Size/PyObject_Length semantics for VariableTracker objects.
    Dispatches to sq_length (sequences) or mp_length (mappings) depending on the VT type.
    """

    if vt_implements_sq_length(obj):
        return obj.sq_length(tx)
    return vt_mapping_size(tx, obj)
