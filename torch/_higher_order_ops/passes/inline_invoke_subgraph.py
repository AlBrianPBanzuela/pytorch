import operator
from typing import Any, TYPE_CHECKING

import torch
from torch.fx.graph_module import GraphModule


if TYPE_CHECKING:
    from torch.fx.node import Node


def inline_single_use_invoke_subgraph(gm: GraphModule) -> GraphModule:
    """Inline invoke_subgraph HOPs whose subgraph is referenced exactly once.

    When a subgraph has only a single caller, invoke_subgraph adds overhead
    without any deduplication benefit, so we inline it unconditionally.
    """
    invoke_nodes = list(
        gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        )
    )
    if not invoke_nodes:
        return gm

    # Recursively apply to nested subgraph modules first.
    for name, mod in gm.named_modules():
        if name and isinstance(mod, GraphModule):
            inline_single_use_invoke_subgraph(mod)

    # Count how many invoke_subgraph nodes reference each subgraph (by get_attr target).
    subgraph_use_count: dict[str, int] = {}
    for node in invoke_nodes:
        target = str(node.args[0].target)  # pyrefly: ignore[missing-attribute]
        subgraph_use_count[target] = subgraph_use_count.get(target, 0) + 1

    single_use_nodes = [
        node
        for node in invoke_nodes
        if subgraph_use_count[str(node.args[0].target)]
        == 1  # pyrefly: ignore[missing-attribute]
    ]
    if not single_use_nodes:
        return gm

    inline_invoke_subgraph_nodes(gm, single_use_nodes)
    return gm


def inline_invoke_subgraph_nodes(gm: GraphModule, invoke_nodes: list["Node"]) -> None:
    """Shared helper that inlines a list of invoke_subgraph nodes."""
    for node in invoke_nodes:
        get_attr_node: torch.fx.Node = node.args[0]  # pyrefly: ignore[bad-assignment]
        operands = node.args[2:]

        subgraph: GraphModule = getattr(gm, str(get_attr_node.target))

        env: dict[Node, Any] = dict(
            zip(subgraph.graph.find_nodes(op="placeholder"), operands)
        )

        with gm.graph.inserting_before(node):
            for sub_node in subgraph.graph.nodes:
                if sub_node.op in ("placeholder", "output"):
                    continue
                env[sub_node] = gm.graph.node_copy(sub_node, lambda n: env[n])

        output_values = subgraph.graph.output_node().args[0]

        for user in list(node.users):
            if user.op == "call_function" and user.target is operator.getitem:
                idx = user.args[1]
                user.replace_all_uses_with(env[output_values[idx]])  # pyrefly: ignore
                gm.graph.erase_node(user)

        gm.graph.erase_node(node)

        if not get_attr_node.users:
            gm.graph.erase_node(get_attr_node)


def inline_invoke_subgraph(gm: GraphModule) -> GraphModule:
    """Inline all invoke_subgraph HOPs, producing a flat FX graph.

    This is useful when downstream compilers (like vllm-compile) don't support
    HOPs or prefer a flat graph, but we still want the Dynamo tracing-time
    benefits of auto-caching (trace once, stamp out cached calls).
    """
    invoke_nodes = list(
        gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        )
    )
    if not invoke_nodes:
        return gm

    # Recursively inline any nested invoke_subgraph calls inside
    # subgraph modules themselves.
    for name, mod in gm.named_modules():
        if name and isinstance(mod, GraphModule):
            inline_invoke_subgraph(mod)

    inline_invoke_subgraph_nodes(gm, invoke_nodes)
    return gm
