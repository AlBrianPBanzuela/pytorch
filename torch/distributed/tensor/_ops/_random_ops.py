# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from torch._ops import OpOverload
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import ArgsType, KwargsType
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor.placement_types import Placement


aten = torch.ops.aten


@register_single_dim_strategy(
    [
        aten.normal_.default,
        aten.uniform_.default,
        aten.bernoulli_.float,
        aten.bernoulli.default,
        aten.bernoulli.p,
    ]
)
def random_op_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    self_meta = args_schema[0]
    if not isinstance(self_meta, TensorMeta):
        raise AssertionError(f"Expect TensorMeta but got {type(self_meta)}")
    return [
        [_ShardingPlaceholder(d), _ShardingPlaceholder(d)]
        for d in range(len(self_meta.shape))
    ]


@register_single_dim_strategy(
    [
        aten.bernoulli_.Tensor,
        aten.bernoulli.Tensor,
    ]
)
def random_op_with_p_tensor_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    self_meta = args_schema[0]
    if not isinstance(self_meta, TensorMeta):
        raise AssertionError(f"Expect TensorMeta but got {type(self_meta)}")
    return [
        [_ShardingPlaceholder(d), _ShardingPlaceholder(d), _ShardingPlaceholder(d)]
        for d in range(len(self_meta.shape))
    ]


@register_single_dim_strategy(aten.native_dropout.default)
def native_dropout_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    self_meta = args_schema[0]
    if not isinstance(self_meta, TensorMeta):
        raise AssertionError(f"Expect TensorMeta but got {type(self_meta)}")
    return [
        [_ShardingPlaceholder(d), _ShardingPlaceholder(d), _ShardingPlaceholder(d)]
        for d in range(len(self_meta.shape))
    ]
