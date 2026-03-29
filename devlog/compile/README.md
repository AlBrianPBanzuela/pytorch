# PyTorch Compile DevLog

Technical notes from PyTorch compiler developers — design deep-dives,
performance analyses, directional proposals, and lessons learned.

## Why DevLog?

- **Lower bar than pytorch.org/blog.**  If you can write a good Workplace
  post, you can write a devlog entry.  No editorial pipeline, no marketing
  review — just a PR.
- **AI-accessible.**  LLM coding assistants index the repo and can surface
  these notes as context.
- **OSS-friendly.**  Most compiler discussions have no reason to stay
  internal.
- **Durable.**  Workplace posts fade in days.  Markdown in the repo is
  permanent.

## Topics

| Directory | Scope |
|-----------|-------|
| [`dynamo/`](./dynamo/) | TorchDynamo — bytecode capture, graph tracing, guards |
| [`inductor/`](./inductor/) | TorchInductor — compiler backend, codegen, fusion |
| [`dynamic_shapes/`](./dynamic_shapes/) | Symbolic reasoning, backed/unbacked, shape guards |

## How to contribute

1. Create a Markdown file: `devlog/compile/<topic>/YYYY-MM-DD-short-title.md`
2. Use [`_template.md`](./_template.md) as a starting point.
3. Open a PR.  Reviewers check technical accuracy and readability.
4. Merge.  Done.

See [`proposal.md`](./proposal.md) for full rationale.
