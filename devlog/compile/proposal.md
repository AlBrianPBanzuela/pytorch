# Proposal: PyTorch DevLog

## TL;DR

Create a `devlog/` directory in the PyTorch repo where developers publish
technical notes as Markdown files via normal PRs.  Organized per
team/subsystem, with a deliberately lower bar than the official
pytorch.org/blog, and automatically accessible to AI coding assistants and
OSS contributors.

## Problem

PyTorch developers regularly write deep technical content — design
discussions, performance analyses, directional proposals, feature
introductions — and share them as ephemeral posts.  This content has a
practical lifespan of days to weeks before it is buried under the daily
volume of new posts.  The consequences:

- **Knowledge is lost.**  A post explaining FSDP2 memory management,
  TorchDynamo guard specialization, or export graph serialization becomes
  undiscoverable within a month.
- **OSS contributors never see it.**  Most of these discussions are purely
  technical with no reason to stay internal, yet they never reach the
  open-source community.
- **AI coding assistants can't use it.**  LLM tools that index the PyTorch
  repo (Copilot, etc.) have no way to learn from content that isn't
  checked in.

## Proposal

Add a `devlog/` directory to the PyTorch repository, open to any team:

```
pytorch/
└── devlog/
    ├── README.md
    ├── compile/
    │   ├── dynamo/
    │   ├── inductor/
    │   └── dynamic_shapes/
    ├── distributed/
    │   ├── fsdp/
    │   ├── dtensor/
    │   └── c10d/
    ├── export/
    ├── core/
    │   ├── autograd/
    │   └── dispatcher/
    └── ...
```

Posts are plain Markdown files following a simple naming convention
(`YYYY-MM-DD-short-title.md`).  Publishing is just a PR.  To start a new
area, create `devlog/<your-area>/README.md` and open a PR.

## Why "DevLog" and not just "Blog"?

The name is intentional.

**"DevLog"** — signals that this is not the official pytorch.org/blog:

- **Lower bar.**  A devlog entry can be a rough technical note, a design
  sketch, a benchmark report, or a postmortem.  It does not need to be
  publication-quality prose.  If you can write a good technical post, you
  can write a devlog entry.
- **Developer audience.**  Written by developers, for developers.  Jargon
  is fine.  Code snippets are encouraged.  No marketing review needed.
- **Living documents.**  Entries can be updated via follow-up PRs as
  designs evolve — with full revision history.

## How DevLog differs from pytorch.org/blog

The official [PyTorch Blog](https://pytorch.org/blog/) goes through an
editorial pipeline: draft in Google Docs → technical reviewers →
comms/legal review → marketing request → marketing team publishes.  This
process takes weeks and produces polished, broad-audience content.

The PyTorch DevLog is **complementary**, not a replacement:

| Aspect              | pytorch.org/blog                   | PyTorch DevLog                       |
|---------------------|------------------------------------|--------------------------------------|
| **Bar to publish**  | High — polished prose, editorial review, marketing sign-off | Low — technically accurate Markdown, normal code review |
| **Audience**        | Broad community, newcomers         | Developers, contributors, power users |
| **Tone**            | Polished, introductory             | Technical, detailed, can be rough    |
| **Review process**  | Editorial + marketing              | Normal PR review                     |
| **Time to publish** | Weeks                              | Same day                             |
| **Organization**    | Single chronological feed          | Per-team/subsystem directories       |
| **AI indexing**     | Not in repo — invisible to agents  | In repo — automatically indexed      |
| **Discoverability** | Web search, RSS                    | GitHub search, AI agents, README indexes |

Think of the devlog as the **working notebook** and pytorch.org/blog as
the **published paper**.  A devlog entry that gains traction can later be
polished into a pytorch.org/blog article.

## Benefits

### 1. AI-accessible context
LLM coding assistants already index the PyTorch repo.  DevLog entries
become part of the context these tools draw on when answering questions
about PyTorch internals — no special integration needed.

### 2. OSS visibility
The PyTorch community extends well beyond Meta.  Technical write-ups about
FSDP internals, Dynamo tracing, export serialization, or autograd edge
cases are exactly what external contributors need.  Hosting them in-repo
makes them discoverable via GitHub search, linkable from issues and PRs,
and citable in external discussions.

### 3. Durability and organization
Markdown files in the repo are versioned, searchable, and permanent.
Ephemeral posts fade within days.  Per-subsystem directories make it
trivial to answer "what's new with FSDP?" or "what's new with dynamic
shapes?" — just check the relevant README.

### 4. Low-friction publishing
No editorial pipeline.  No marketing review.  Write Markdown, open a PR,
get a code-review-level approval, merge.  This removes the activation
energy that prevents most technical content from ever being written down
permanently.

### 5. Review and iteration
The PR workflow means entries get technical review before merging, and can
be updated later as designs evolve — without losing revision history.

## What belongs in DevLog

- **Design deep-dives** — "How FSDP2 memory management works"
- **Performance analyses** — "Unbacked vs. backed dynamic shapes in vLLM:
  a 29-model benchmark"
- **Directional proposals** — "Towards zero-copy DTensor resharding"
- **Feature introductions** — "What is `torch.export` and when should you
  use it?"
- **Postmortems / lessons learned** — "Debugging a 4% Mixtral regression
  from symbolic shape guards"
- **Migration guides** — "Migrating from FSDP1 to FSDP2"

Content that is confidential, business-sensitive, or not technically useful
to the OSS community should remain internal.

## Process

1. Create a Markdown file using the provided `_template.md`.
2. Place it in the appropriate topic directory
   (e.g., `devlog/compile/dynamo/`, `devlog/distributed/fsdp/`).
3. Open a PR — reviewers check technical accuracy and readability.
4. Merge.  Done.

No build system changes.  No website deployment.  No CI integration.  Just
Markdown files in a directory.

## Why plain Markdown instead of GitHub Pages?

GitHub Pages (Jekyll, Hugo, MkDocs) would generate a polished website.  We
intentionally chose plain Markdown:

1. **AI indexing is the primary goal.**  LLM agents index repo files.  They
   do *not* index rendered HTML from a `gh-pages` branch.  A GitHub Pages
   site would defeat the main purpose.
2. **pytorch.org/blog already exists.**  A second website would create
   confusion.  Plain Markdown in the main repo occupies a clearly different
   niche.
3. **Zero maintenance.**  No static-site generator, no CI pipeline, no
   theme updates.  Just Markdown.
4. **GitHub renders it well.**  The reading experience on GitHub is already
   good — no build step needed.
5. **Upgrade path is open.**  If we later want a rendered site or RSS feed,
   we can add a build layer without changing existing entries.

**Commenting:**  Technical discussion happens during PR review (before
merge) and in GitHub Issues or Discussions (after merge).  This is the same
model used by Linux kernel docs, Rust RFCs, and Python PEPs.

## Example posts (already converted)

The compile team has seeded five posts under `compile/dynamic_shapes/`:

| Date | Title |
|------|-------|
| 2026-03-25 | [Unbacked Dynamic Shapes Shouldn't Be Slower — Now They Aren't](dynamic_shapes/2026-03-25-unbacked-perf-parity.md) |
| 2026-02-27 | [Reducing Compile-Time Overhead in Unbacked-Symbol-Heavy Export Traces](dynamic_shapes/2026-02-27-compile-time-unbacked-export.md) |
| 2026-01-20 | [Backed to Unbacked: From Guardable to Guardless Shapes](dynamic_shapes/2026-01-20-backed-to-unbacked.md) |
| 2025-10-29 | [Slaying the Framework Data-Dependent Errors Dragon](dynamic_shapes/2025-10-29-slaying-framework-ddes.md) |
| 2025-07-08 | [Guard-Free Dynamic Shapes](dynamic_shapes/2025-07-08-guard-free-dynamic-shapes.md) |

## Next steps

1. Land the initial directory structure and seed posts.
2. Announce and encourage contributions from all PyTorch teams.
