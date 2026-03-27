#!/usr/bin/env python3
"""
Rerun PyTorch CI jobs via blast Remote Execution.

Usage:
  # Dry run to see the generated script:
  python re_demo.py run .github/workflows/lint-osdc.yml -j lintrunner-noclang --dry-run

  # Run with a specific PR:
  python re_demo.py run .github/workflows/lint-osdc.yml -j quick-checks --pr 178000

  # Override the script (when it has GHA expressions):
  python re_demo.py run .github/workflows/lint-osdc.yml -j lintrunner-noclang --pr 178000 \\
    --cmd 'ADDITIONAL_LINTRUNNER_ARGS="--all-files" .github/scripts/lintrunner.sh'

  # Quick run without workflow parsing:
  python re_demo.py run --cmd 'echo hello' --pr 178000

  # List jobs in a workflow:
  python re_demo.py list .github/workflows/lint-osdc.yml
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def _ensure_deps():
    try:
        import re_cli  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "blast-cli", "pyyaml",
        ])

_ensure_deps()

import yaml  # noqa: E402
from re_cli.core.core_types import StepConfig  # noqa: E402
from re_cli.core.job_runner import JobRunner  # noqa: E402
from re_cli.core.k8s_client import K8sClient, K8sConfig  # noqa: E402
from re_cli.core.script_builder import RunnerScriptBuilder  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO = "https://github.com/pytorch/pytorch.git"
REPO_LOCAL = str(Path(__file__).resolve().parent.parent.parent)
IMAGE = "ghcr.io/pytorch/test-infra:cpu-x86_64-67eb930"
GHA_EXPR = re.compile(r"\$\{\{.*?\}\}")

# ---------------------------------------------------------------------------
# Script builder
# ---------------------------------------------------------------------------

class PyTorchScriptBuilder(RunnerScriptBuilder):
    DEFAULT_MODULES = [
        "header", "find_script", "git_clone",
        "git_checkout", "run_script", "upload_outputs",
    ]

    def add_git_clone(self) -> "PyTorchScriptBuilder":
        self._modules.append(
            f"\n# {'=' * 44}\n# MODULE: git_clone\n# {'=' * 44}\n"
            'if [[ -n "$GIT_REPO" ]]; then\n'
            '    echo "[Runner] Cloning $GIT_REPO..."\n'
            '    git clone --depth=1 "$GIT_REPO" repo\n'
            '    cd repo\n'
            '    REPO_DIR="$(pwd)"\n'
            '    export REPO_DIR\n'
            '    echo "[Runner] REPO_DIR=$REPO_DIR"\n'
            "else\n"
            '    echo "[Runner] No git repo specified, skipping clone"\n'
            "fi\n"
        )
        return self

# ---------------------------------------------------------------------------
# Action → bash mapping
# ---------------------------------------------------------------------------

ACTIONS_DIR = Path(__file__).resolve().parent / "actions"


def _action_setup_uv(inputs: dict) -> str:
    """Inline setup_uv.sh with inputs passed as env vars."""
    template = (ACTIONS_DIR / "setup_uv.sh").read_text()
    # Strip shebang/comments/set lines — outer script handles those
    body = "\n".join(
        line for line in template.splitlines()
        if not line.startswith("#") and line.strip() != "set -eu"
    ).strip()
    # Set env vars from action inputs before the script body
    env = {
        "PYTHON_VERSION": str(inputs.get("python-version", "3.12")),
        "ACTIVATE_ENV": str(inputs.get("activate-environment", "false")).lower(),
        "UV_VERSION": str(inputs.get("uv-version", "0.9.21")),
    }
    exports = "\n".join(f'export {k}="{v}"' for k, v in env.items())
    return f"{exports}\n{body}"


ACTION_MAP = {
    "pytorch/test-infra/.github/actions/setup-uv": _action_setup_uv,
}

# ---------------------------------------------------------------------------
# Workflow parsing
# ---------------------------------------------------------------------------

def _find_repo_root(from_path: str) -> Path:
    p = Path(from_path).resolve().parent
    while p != p.parent:
        if (p / ".git").exists() or (p / ".github").exists():
            return p
        p = p.parent
    return Path(from_path).resolve().parent


def extract_setup_steps(uses: str, caller_path: str) -> list[dict]:
    """Extract setup steps from a reusable workflow.

    - run: steps with `# re:add` → include bash directly
    - uses: steps in ACTION_MAP → convert to bash
    """
    if not uses.startswith("./"):
        return []
    root = _find_repo_root(caller_path)
    resolved = root / uses.split("@")[0].removeprefix("./")
    if not resolved.exists():
        return []

    wf = yaml.safe_load(resolved.read_text())
    substeps = []
    for _job_name, job_def in wf.get("jobs", {}).items():
        for step in job_def.get("steps", []):
            name = step.get("name", f"step-{len(substeps)}")

            run = step.get("run")
            if run and "# re:add" in run:
                substeps.append({"name": name, "bash": run.strip()})
                continue

            action_key = step.get("uses", "").split("@")[0]
            if action_key in ACTION_MAP:
                bash = ACTION_MAP[action_key](step.get("with", {}))
                substeps.append({"name": name, "bash": bash})

    return substeps


def parse_workflow_jobs(path: str) -> dict[str, dict]:
    """Parse jobs with `uses:` + `with.script` from a workflow file."""
    wf = yaml.safe_load(Path(path).read_text())
    jobs = {}
    for name, defn in wf.get("jobs", {}).items():
        uses = defn.get("uses", "")
        if not uses:
            continue
        w = defn.get("with", {})
        script = w.get("script", "")
        if not script:
            continue
        jobs[name] = {
            "image": w.get("docker-image", ""),
            "script": script,
            "uses": uses,
            "has_gha_expr": bool(GHA_EXPR.search(script)),
        }
    return jobs


def build_command(setup_steps: list[dict], cmd: str) -> str:
    parts = ["#!/bin/bash", "set -e", ""]
    for s in setup_steps:
        parts.append(f"# === {s['name']} ===")
        parts.append(s["bash"].strip())
        parts.append("")
    parts.append("# === run ===")
    parts.append(cmd.strip())
    parts.append("")
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# PR / commit resolution
# ---------------------------------------------------------------------------
def pr_info(pr: int) -> dict:
    out = subprocess.run(
        ["gh", "pr", "view", str(pr), "--repo", "pytorch/pytorch",
         "--json", "headRefOid,headRefName,headRepository,headRepositoryOwner"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    owner = data["headRepositoryOwner"]["login"]
    repo_name = data["headRepository"]["name"]
    return {
        "sha": data["headRefOid"],
        "branch": data["headRefName"],
        "repo": f"https://github.com/{owner}/{repo_name}.git",
    }


def resolve(args) -> dict:
    if args.pr:
        info = pr_info(args.pr)
        print(f"PR #{args.pr} -> {info['sha'][:12]} ({info['repo']})")
        return {"sha": info["sha"], "repo": info["repo"]}
    if args.commit:
        return {"sha": args.commit, "repo": REPO}
    print("Provide --pr or --commit")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def submit(steps: list[StepConfig], name: str, args):
    client = K8sClient(K8sConfig(namespace="remote-execution-system", timeout=60))
    resolved = resolve(args)
    runner = JobRunner(
        client=client, name=name, step_configs=steps,
        script_builder_class=PyTorchScriptBuilder,
    )
    runner.run(
        commit=resolved["sha"],
        repo=resolved["repo"],
        follow=not args.no_follow,
        dry_run=args.dry_run,
    )
    if runner.run_id:
        print(f"\nRun ID: {runner.run_id}")
        print(f"Stream:  blast stream {runner.run_id}")

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_run(args):
    setup_steps = []
    image = args.image or IMAGE
    cmd = args.cmd

    if args.workflow and args.job:
        jobs = parse_workflow_jobs(args.workflow)
        if args.job not in jobs:
            print(f"Job '{args.job}' not found. Available: {', '.join(jobs)}")
            sys.exit(1)

        job = jobs[args.job]
        image = args.image or job["image"] or IMAGE
        setup_steps = extract_setup_steps(job["uses"], args.workflow)

        if not cmd:
            if job["has_gha_expr"]:
                print("Script contains GHA expressions that can't run outside CI:\n")
                print("  " + job["script"].replace("\n", "\n  "))
                print("\nProvide --cmd with your adapted command. Example:")
                clean = GHA_EXPR.sub('"*"', job["script"])
                print(f"  --cmd '{clean.strip().splitlines()[-1].strip()}'")
                sys.exit(1)
            cmd = job["script"]
    elif args.workflow:
        print("--workflow requires --job / -j")
        sys.exit(1)

    if not cmd:
        print("Provide --cmd or --workflow + --job")
        sys.exit(1)

    command = build_command(setup_steps, cmd)
    step = StepConfig(
        name=args.job or "run",
        command=command,
        task_type="cpu-large",
        image=image,
    )
    job_name = f"{step.name}-pr{args.pr}" if args.pr else step.name
    submit([step], job_name, args)


def cmd_list(args):
    jobs = parse_workflow_jobs(args.workflow)
    if not jobs:
        print(f"No runnable jobs in {args.workflow}")
        sys.exit(1)

    print(f"Jobs in {args.workflow}:\n")
    for name, info in jobs.items():
        tag = info["image"].split(":")[-1][:30] if info["image"] else "-"
        dynamic = " (has ${{}})" if info["has_gha_expr"] else ""
        print(f"  {name:<35} {tag}{dynamic}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("run", help="Run a CI job on RE")
    r.add_argument("workflow", nargs="?", help="Path to workflow YAML")
    r.add_argument("-j", "--job", help="Job name from the workflow")
    r.add_argument("--cmd", help="Command to run (overrides workflow script)")
    r.add_argument("--image", help="Override Docker image")
    r.add_argument("--pr", type=int, help="PR number")
    r.add_argument("--commit", help="Commit SHA")
    r.add_argument("--patch", action="store_true", help="Include local changes")
    r.add_argument("--dry-run", action="store_true", help="Show what would run")
    r.add_argument("--no-follow", action="store_true", help="Don't stream logs")
    r.set_defaults(func=cmd_run)

    ls = sub.add_parser("list", help="List jobs in a workflow")
    ls.add_argument("workflow", help="Path to workflow YAML")
    ls.set_defaults(func=cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
