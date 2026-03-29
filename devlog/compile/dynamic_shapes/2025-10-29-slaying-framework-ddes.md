---
title: "Slaying the Framework Data-Dependent Errors Dragon"
author: Laith Sakka (@laithsakka)
date: 2025-10-29
tags: [dynamic_shapes, unbacked, dde, export, guard_free, torch.compile]
---

# Slaying the Framework Data-Dependent Errors Dragon

> **TL;DR** – Framework DDE dragon has been slain!  We've eliminated the
> vast majority of framework data-dependent errors — reducing user issues
> by over **85%** — and unlocked **specialization-free full graph capture**
> that *just works*.  This lays the groundwork for emerging unbacked use
> cases in vLLM, MoE graphs, and Frontier.

## Background

Six months ago, we launched an initiative to eliminate framework DDEs by
implementing **explicit unbacked semantics** — explicitly defining how
code should behave when inputs are unbacked
([see the original post](./2025-07-08-guard-free-dynamic-shapes.md)).

That work is now complete.  We've moved into the maintenance phase, and
many previously error-prone operations — reshaping, slicing, narrowing,
selection, contiguity checks, broadcasting checks — are now **fully
DDE-free**.  In total we addressed more than 270+ code branches.  The old,
complex `guard_size_oblivious` / `size-like` mechanism has been
**completely deprecated**.

This marks a major milestone: we can now **capture specialization-free
graphs** much more reliably, providing a smoother and more predictable
user experience.

## What this means for users and developers

### 1. Improved user experience

- Reports of DDEs from PyTorch users have dropped by **85%** (from 35+ per
  half to just 5).
- We **closed 30+ GitHub issues** related to DDEs — many were no longer
  reproducible, while others involved outdated workarounds that this work
  already resolved.
- A study of 50 open-source models identified DDEs as the **dominant
  exportability issue**.  By the end of the initiative, we eliminated the
  remaining framework-related DDEs in those models — making exporting
  these models much simpler and faster.
- Exporting complex models now **saves two or more weeks** of development
  time per model.

### 2. Reduced technical complexity

The previous `guard_size_oblivious` / `size-like` system was the first
step toward eliminating DDEs, but it introduced multiple layers of
overhead:

- **Size-like annotation and propagation** — users had to manually call
  `_check_size()` to mark size-like dimensions, and the framework had to
  correctly propagate those annotations across operations.  Any missed
  annotation or propagation failure broke the system's guarantees.
- **Dependence on symbolic reasoning** — the system relied on a hint-free
  symbolic evaluator to infer relationships among dynamic shapes.  If
  inference failed or remained incomplete, DDEs would persist.

With **explicit unbacked semantics**, all of this complexity has been
removed: no manual `_check_size()` calls, no propagation of "size-
likeness," no reliance on symbolic evaluation.  The result is a **simpler,
more deterministic, and more predictable system** that achieves better DDE
elimination.

### 3. Enabling sound, non-constrained graphs

In the past, users often had to insert `torch._check` calls to constrain
the graph and avoid DDEs — then manually remove or ignore those checks
later to generalize exported graphs.  It was a fragile and frustrating
workaround.

With unbacked semantics, that's no longer necessary.  Users can now produce
**fully general, unconstrained graphs** directly — without resorting to
manual hacks.

## What's next for unbacked dynamic shapes

Support for unbacked dynamic shapes remains a key theme — especially as
their importance grows with deterministic compilation, compile-on-one-rank,
Frontier, and vLLM soundness.  Key remaining areas:

- **Improve performance of unbacked shapes** to match backed dynamic
  shapes (initially using vLLM as a proxy).
- **Improve size hinting consistency** by making size hinting for unbacked
  symbols consistent with the shape environment, and allow users to
  control their hints.
- **Ensure unbacked shapes are guard-free** unless explicitly requested by
  users, and introduce unbacked striding policies for layout properties.
- **Build infrastructure to support pre-compile API** — hooks that allow
  users to mark dynamism and provide invariants without changing model
  code.
- **Address remaining less-frequent DDE sources** — a few GitHub issues
  remain open, and we continue monitoring reports and using the fuzzer to
  uncover unhandled call sites.
- **Enable unbacked shapes for model inputs in export API.**
- **Harden runtime assertion lowering** by ensuring assertions are added
  when expected.

## References

- [Guard-Free Dynamic Shapes — original initiative post](./2025-07-08-guard-free-dynamic-shapes.md)
- [`torch/fx/experimental/symbolic_shapes.py`](../../../torch/fx/experimental/symbolic_shapes.py) — symbolic shape infrastructure
- [Backed to Unbacked — broader context](./2026-01-20-backed-to-unbacked.md)
