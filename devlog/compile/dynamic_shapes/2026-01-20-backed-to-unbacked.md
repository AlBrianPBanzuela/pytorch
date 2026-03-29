---
title: "Backed to Unbacked: From Guardable to Guardless Shapes in PyTorch"
author: Laith Sakka (@laithsakka), Aditya Venkataraman, Bob Ren
date: 2026-01-20
tags: [dynamic_shapes, unbacked, backed, torch.compile, frontier]
---

# Backed to Unbacked: From Guardable to Guardless Shapes in PyTorch

![Hint stamp](./images/2026-01-20-backed-unbacked-header.jpg)

> **TL;DR**

> **TL;DR** – We expect unbacked dynamic shapes to become the dominant
> shape mechanism for Frontier-style workloads due to their better
> predictability and controllability.  However, some blockers remain —
> most notably the performance gap, which is a primary focus for the first
> half of 2026.

## 1. The Emergence of Backed Shapes

### 1.1 Endless recompilations

We created PT2 as a JIT drop-in, plug-and-play method to accelerate ML
programs — simply by adding a decorator.  People started using
`torch.compile` and life was good — until we began hitting recompilations
that were killing performance for some important use cases.

Consider the following simple function:

```python
def func(x):
    return torch.ones(x)
```

When compiled with `x=10`, Dynamo inserted a guard checking that "the
input `x` is exactly 10" and generated a graph hard-coded to return a
tensor of size 10.  Next, when called with `x=20`, the guard failed,
triggering another compilation for size 20, and so on.

For workloads with many varying inputs, this led to endless recompilations
and became a serious performance problem.  So, what did we do?  We stopped
hard-coding (specializing) sizes in the graph and made them dynamic by
representing them symbolically:

```python
def func_symbolic(s0):
    return torch.ones(s0)
```

### 1.2 Back it with a hint

Of course, it wasn't that simple.  Once we represented sizes symbolically,
the compiler encountered issues whenever it needed to branch based on
those sizes.  Branching could occur in framework/compiler code (e.g.,
contiguity checks) or in user code:

```python
if x < 1024:
    # path A
else:
    # path B
```

With symbolic shapes, the compiler no longer had a concrete value for `x`
to decide which path to take during compilation.  To solve this, we
**backed** the dynamic shape (`s0`) with a **hint** — a concrete value
from the example input used during compilation.

For instance, if the example value at compile time was 10, then we use 10
as the hint for `x`.  During compilation, `path A` is picked based on the
hint, and Dynamo then adds a guard ensuring that the condition of the
taken branch (`x < 1024`) is satisfied.

This approach effectively solved the "endless recompilations" problem;
instead of recompiling for each new concrete value, we only compile once
for each branch taken.

We named these **backed dynamic shapes** because they are backed by a
hint that guides branch selection.  Another term is **guardable shapes**,
as we are allowed to introduce guards that constrain them within the
Dynamo graph.

## 2. The Emergence of Unbacked Shapes

### 2.1 Data-dependent ops

In a different use case, we encountered functions that use data-dependent
operations:

```python
def func(x):
    u = x.item()
```

Here, `u` is a scalar value that depends on the data of `x`.  At compile
time, we do not know the concrete value of `u`.

Initially, the trivial option was to give up and trigger a **graph break**
— force the compilation to stop, split the compiled graph into two, and
execute the data-dependent operation eagerly (`.item()` here) to get a
concrete value for `u`.  This would then resume compiling the second graph
with a known integer input.

This was problematic for export and other use cases requiring a single,
full graph.  Furthermore, data-dependent-heavy code would result in many
graph breaks, hurting performance and significantly increasing compile
time.

### 2.2 Unbacked dynamic shapes

To keep the data-dependent operation within the graph without graph
breaking, we represented its output symbolically — with a different type
of shape.  Unlike backed shapes, we do not have a hint to use for
resolving branching on `u`.  That's why we call these **unbacked dynamic
shapes**.

A significant challenge was handling branching without the hint; without
a hint, the compiler couldn't determine which branch to take, and the
default behavior was to **throw a data-dependent error (DDE)**.  For
example:

```python
def func(x):
    u = x.item()
    y = .. if u == 0 else ..
```

This was one of the most painful UX aspects of dynamic shapes.  We
addressed this by teaching framework code how to handle these branches —
automatically picking the general path — and by providing APIs for users
to write DDE-friendly branching.  This work resolved the single most
common reason for export failures in the framework.

### 2.3 Unbacked inputs

While unbacked shapes were originally introduced for data-dependent
operations, over time users began deliberately choosing unbacked shapes
for primary graph inputs as well (effectively dropping the hints and
treating those inputs as if they came from data-dependent ops).

The main motivations were twofold:

1. **Avoid branch-induced recompilations** — with unbacked shapes, general
   branches valid for all inputs are selected without imposing shape
   constraints.  This ensures no recompilations occur ever.
2. **Compile graphs that work across all input shapes** — a single compiled
   graph can handle a broad range of input shapes.

Unbacked shapes can also be referred to as **guardless dynamic shapes**.
Not only do they lack a "hint for the purpose of guarding", but they are
also not allowed to have guards.

> Note: unbacked dynamic shapes *can* have hints in the form of
> "optimization hints", but those hints can only be used for guardless
> optimizations such as auto-tuning.

## 3. Unbacked is a Better Fit for the Frontier

### 3.1 Predictability, determinism, and control

Backed dynamic shapes — while a strong fit for drop-in JIT optimization —
conflict with Frontier's focus on determinism, predictability, and
control.  This is precisely where unbacked shapes excel:

- **Predictability.**  Backed shapes are difficult to reason about ahead
  of time.  You cannot know what constraints will be imposed on dynamic
  input ranges without actually compiling, nor is it clear which example
  inputs are required to generate enough graphs to cover the full input
  space.  By contrast, unbacked shapes are highly predictable.  Users can
  explicitly request: "Compile one graph with `x` unconstrained and
  another with `x` in `[0, 100]`."  If compilation succeeds, you know
  the compiler has not introduced any additional, hidden constraints.

- **Determinism.**  The output graph for backed dynamic shapes depends
  heavily on the specific example inputs used during compilation, and is
  very sensitive to source changes (including added or removed
  optimizations), since different branches may or may not be taken.

- **Pre-compile.**  This creates challenges for pre-compilation since we
  want to ensure that a reasonable, finite set of graphs collectively
  covers the entire relevant input space.  With backed shapes, the
  compiler's implicit constraints and dependence on example inputs make it
  difficult to know whether the precompiled graphs truly cover all
  intended inputs.

A concrete example is the vLLM use case, where multiple graphs are
precompiled for different input ranges and are expected to work reliably
within those specified ranges.  Backed shapes are **fundamentally unsound
here** — leading to continuous issues because the compiler can silently
introduce restrictive shape constraints.  Unbacked shapes, by design, are
the correct tool for this scenario.

### 3.2 Are backed about to die?

Not at all.  While Frontier clearly points toward unbacked shapes, backed
shapes are far from obsolete.  For the typical PyTorch user who just wants
a drop-in JIT optimization, backed dynamic shapes remain an excellent
choice: they provide strong performance benefits without requiring deep
model understanding or the overhead of manually compiling and managing
multiple unbacked graphs.

### 3.3 Remaining blockers for Frontier's unbacked shapes

There are three major issues that must be addressed:

1. **Framework DDEs** — resolved in the last year.
2. **Performance** — unbacked shapes had ~30% performance drops on vLLM
   models and up to 85% regressions on TorchBench HuggingFace models.
   Closing this gap is a key goal for the first half of 2026.
3. **Expressive dispatch** — APIs that allow users to compile multiple
   unbacked graphs with explicit shape constraints and automatically
   dispatch among them, especially in pre-compilation workflows.

## References

- [`torch/fx/experimental/symbolic_shapes.py`](../../../torch/fx/experimental/symbolic_shapes.py) — symbolic shape infrastructure
- [`torch/_dynamo/variables/builder.py`](../../../torch/_dynamo/variables/builder.py) — unbacked symbol creation
