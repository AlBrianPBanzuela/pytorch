#!/usr/bin/env python3
"""
CLAUDE_REVIEW: AI-powered code review using Claude on git diffs.

This is a lintrunner adapter that sends your git diff to Claude for review.
It auto-gates: silently exits in CI or when the `claude` CLI is not installed.
To skip: lintrunner --skip CLAUDE_REVIEW
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from enum import Enum
from typing import NamedTuple


LINTER_CODE = "CLAUDE_REVIEW"
MAX_PROMPT_CHARS = 150_000
LOG = logging.getLogger(__name__)

# ── Gate: skip in CI ─────────────────────────────────────────────────────
CI_ENV_VARS = ("CI", "GITHUB_ACTIONS", "CIRCLECI", "JENKINS_URL", "TRAVIS")


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def is_ci() -> bool:
    """Return True if running inside a CI environment."""
    return any(os.environ.get(v) for v in CI_ENV_VARS)


def get_merge_base() -> str:
    """Get the merge base commit to diff against."""
    for remote_branch in ("origin/main", "origin/master"):
        try:
            result = subprocess.run(
                ["git", "merge-base", "HEAD", remote_branch],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    # Fallback: diff against HEAD~1
    return "HEAD~1"


def get_diff_for_files(filenames: list[str]) -> dict[str, str]:
    """Get the git diff for each file, keyed by filename."""
    merge_base = get_merge_base()
    LOG.info("Diffing against merge base: %s", merge_base)

    try:
        result = subprocess.run(
            ["git", "diff", merge_base, "--"] + filenames,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        LOG.warning("Failed to get diff")
        return {}

    diffs: dict[str, str] = {}
    if not result.stdout.strip():
        return diffs

    # Split combined diff output by "diff --git" headers
    current_file = None
    current_lines: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("diff --git "):
            if current_file and current_lines:
                diffs[current_file] = "\n".join(current_lines)
            header = re.match(r"^diff --git a/(.*) b/(.*)$", line)
            current_file = header.group(2) if header else line.split(" b/", 1)[-1]
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_file and current_lines:
        diffs[current_file] = "\n".join(current_lines)

    return diffs


def _get_repo_root() -> str:
    """Get the git repository root directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return os.path.dirname(os.path.abspath(__file__))


def _read_skill_files() -> str | None:
    """Read the pr-review skill files to use as the review prompt base.

    Returns None if no skill files could be read.
    """
    repo_root = _get_repo_root()
    skill_dir = os.path.join(repo_root, ".claude", "skills", "pr-review")

    parts = []
    for filename in ("SKILL.md", "review-checklist.md", "bc-guidelines.md"):
        path = os.path.join(skill_dir, filename)
        try:
            with open(path) as f:
                parts.append(f.read())
        except OSError:
            LOG.warning("Could not read skill file: %s", path)

    if not parts:
        return None
    return "\n\n".join(parts)


# Appended to the skill prompt to request structured JSON output for lintrunner.
_JSON_OUTPUT_FORMAT = """\

Be fast. Only flag clear bugs, correctness issues, or serious problems. \
Skip minor style or readability nits. Limit your response to at most 5 issues.

For each issue, specify the EXACT filename and line number in the NEW version \
of the file (lines starting with +).

Return a JSON array of issues. Each issue must have:
  - "path": string (filename)
  - "line": integer (line number in the new file, or null if file-level)
  - "severity": "error" | "warning" | "advice"
  - "name": string (short title, <60 chars)
  - "description": string (explanation + suggested fix)

If the code looks good, return an empty array: []

Return ONLY the JSON array, no markdown fences, no other text.
"""


def build_review_prompt(diffs: dict[str, str]) -> str | None:
    """Build the Claude prompt from file diffs and the pr-review skill.

    Returns None if the skill files could not be loaded.
    """
    skill_prompt = _read_skill_files()
    if skill_prompt is None:
        LOG.warning(
            "pr-review skill files not found under .claude/skills/pr-review/. "
            "Skipping review."
        )
        return None

    diff_text = "\n\n".join(f"--- {fname} ---\n{diff}" for fname, diff in diffs.items())
    return f"""{skill_prompt}
{_JSON_OUTPUT_FORMAT}
DIFF:
{diff_text}
"""


def get_new_line_numbers(diff_text: str) -> set[int]:
    """Parse a unified diff and return the set of new-file line numbers for added lines."""
    new_lines: set[int] = set()
    in_hunk = False
    new_lineno = 0
    for line in diff_text.splitlines():
        hunk_match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if hunk_match:
            new_lineno = int(hunk_match.group(1))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+"):
            new_lines.add(new_lineno)
            new_lineno += 1
        elif line.startswith("-"):
            pass  # deleted line, doesn't advance new-file counter
        elif line.startswith("\\"):
            pass  # "\ No newline at end of file" marker
        else:
            # context line
            new_lineno += 1
    return new_lines


def call_claude(prompt: str, model: str = "sonnet") -> list[dict]:
    """Call Claude via the claude CLI and return parsed issues."""
    try:
        result = subprocess.run(
            [
                "claude",
                "-p",
                "--model",
                model,
                "--output-format",
                "text",
                "--no-session-persistence",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            LOG.error(
                "claude CLI failed (exit %d): %s",
                result.returncode,
                result.stderr.strip(),
            )
            return []
        raw = result.stdout.strip()
    except FileNotFoundError:
        LOG.warning(
            "claude CLI not found. Install Claude Code:"
            " https://docs.anthropic.com/en/docs/claude-code"
        )
        return []
    except subprocess.TimeoutExpired:
        LOG.warning("claude CLI timed out after 60s")
        return []

    # Try parsing directly first; fall back to bracket extraction if Claude
    # wrapped the JSON in fences or preamble text.
    try:
        issues = json.loads(raw)
        if not isinstance(issues, list):
            LOG.warning("Claude returned non-list response: %s", type(issues))
            return []
        return issues
    except json.JSONDecodeError:
        pass

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end > start:
        try:
            issues = json.loads(raw[start : end + 1])
            if isinstance(issues, list):
                return issues
        except json.JSONDecodeError:
            pass

    LOG.warning("Failed to parse Claude response as JSON. Raw output: %s", raw[:500])
    return []


def review_files(filenames: list[str], model: str = "sonnet") -> list[LintMessage]:
    """Main review logic: get diffs, call Claude, emit LintMessages."""
    diffs = get_diff_for_files(filenames)
    if not diffs:
        LOG.info("No diffs found for the given files, skipping review.")
        return []

    prompt = build_review_prompt(diffs)
    if prompt is None:
        return []

    if len(prompt) > MAX_PROMPT_CHARS:
        LOG.warning(
            "Diff too large (%d chars, max %d). Skipping review.",
            len(prompt),
            MAX_PROMPT_CHARS,
        )
        return []

    LOG.info(
        "Sending %d file diff(s) to Claude for review (%d chars)...",
        len(diffs),
        len(prompt),
    )

    issues = call_claude(prompt, model=model)
    LOG.info("Claude returned %d issue(s).", len(issues))

    # Pre-compute valid new-file line numbers per file for validation
    valid_lines_by_file: dict[str, set[int]] = {
        fname: get_new_line_numbers(diff) for fname, diff in diffs.items()
    }

    severity_map = {
        "error": LintSeverity.ERROR,
        "warning": LintSeverity.WARNING,
        "advice": LintSeverity.ADVICE,
    }

    messages = []
    # Use diff keys as canonical paths (git-relative) and keep insertion order
    valid_files = list(diffs.keys())
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        path = issue.get("path")
        # Validate the path is one of our files; basename fallback only if unambiguous
        if path and path not in valid_lines_by_file:
            target = os.path.basename(path)
            matches = [f for f in valid_files if os.path.basename(f) == target]
            path = matches[0] if len(matches) == 1 else None

        # When path couldn't be resolved, clear line too
        if path is None:
            line = None
        else:
            line = issue.get("line")

        # Drop line if it's not in the diff rather than snapping to unrelated code
        if line is not None and path is not None and path in valid_lines_by_file:
            valid = valid_lines_by_file[path]
            if valid and line not in valid:
                line = None

        severity_str = issue.get("severity", "advice").lower()
        severity = severity_map.get(severity_str, LintSeverity.ADVICE)

        messages.append(
            LintMessage(
                path=path,
                line=line,
                char=None,
                code=LINTER_CODE,
                severity=severity,
                name=issue.get("name", "Claude review comment"),
                original=None,
                replacement=None,
                description=issue.get("description", ""),
            )
        )

    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Claude AI code review linter for lintrunner",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--model",
        default="sonnet",
        help="Claude model to use (default: sonnet)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
    )

    # ── Gate: never run in CI ──
    if is_ci():
        LOG.info("CI environment detected, skipping Claude review.")
        return

    # ── Gate: silently skip if claude CLI is not available ──
    if not shutil.which("claude"):
        return

    messages = review_files(args.filenames, model=args.model)
    for msg in messages:
        print(json.dumps(msg._asdict()), flush=True)


if __name__ == "__main__":
    main()
