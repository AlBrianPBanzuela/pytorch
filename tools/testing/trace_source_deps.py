"""Trace dependencies between torch source files and test files.

Two tracing modes:
  import  - import each test module in a subprocess, record sys.modules (fast)
  coverage - run each test under `coverage run`, parse executed files (slow, precise)

Two subcommands:
  trace - build the mapping JSON
  query - look up affected tests for changed source files

Output is compatible with td_heuristic_profiling.json consumed by the
Profiling heuristic in target determination.
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = REPO_ROOT / "test"


# ---------------------------------------------------------------------------
# Test discovery
# ---------------------------------------------------------------------------

def _discover_python_tests() -> list[str]:
    """Return test names from discover_tests, filtering out C++ tests."""
    # Import inline so the rest of the file stays importable without pytorch on
    # sys.path.
    sys.path.insert(0, str(REPO_ROOT))
    from tools.testing.discover_tests import TESTS

    return [t for t in TESTS if not t.startswith("cpp/")]


def _pattern_to_test_names(pattern: str) -> set[str] | None:
    """If pattern looks like a file path that exists, return the matching test
    name(s) so we can do exact-match filtering instead of regex."""
    pairs = _resolve_test_files([pattern])
    if pairs:
        return {name for name, _ in pairs}
    return None


def _filter_tests(
    tests: list[str],
    include: str | None,
    exclude: str | None,
) -> list[str]:
    if include:
        names = _pattern_to_test_names(include)
        if names is not None:
            tests = [t for t in tests if t in names]
        else:
            pat = re.compile(include)
            tests = [t for t in tests if pat.search(t)]
    if exclude:
        names = _pattern_to_test_names(exclude)
        if names is not None:
            tests = [t for t in tests if t not in names]
        else:
            pat = re.compile(exclude)
            tests = [t for t in tests if not pat.search(t)]
    return tests


def _resolve_path(raw: str) -> Path | None:
    """Resolve a user-provided path (relative to cwd or repo root)."""
    p = Path(raw).resolve()
    if p.exists():
        return p
    p = (REPO_ROOT / raw).resolve()
    if p.exists():
        return p
    return None


def _file_to_test_pair(p: Path) -> tuple[str, str]:
    """Convert an absolute .py file path to a (test_name, abs_path) pair."""
    try:
        rel = p.relative_to(TEST_DIR)
        test_name = str(rel.with_suffix("")).replace(os.sep, "/")
    except ValueError:
        try:
            rel = p.relative_to(REPO_ROOT)
        except ValueError:
            rel = p
        test_name = str(rel.with_suffix("")).replace(os.sep, "/")
    return (test_name, str(p))


def _resolve_test_files(
    file_paths: list[str],
) -> list[tuple[str, str]]:
    """Convert user-provided file/directory paths to (test_name, abs_path) pairs.

    Accepts paths relative to repo root, relative to cwd, or absolute.
    Directories are expanded recursively to all test_*.py files within.
    The test_name is the key used in the output mapping — for files under
    test/ it matches the discover_tests convention (e.g. "test_nn",
    "distributed/test_rpc"); for other files it's the repo-relative path
    without the .py extension.
    """
    pairs: list[tuple[str, str]] = []
    for raw in file_paths:
        p = _resolve_path(raw)
        if p is None:
            print(f"Warning: skipping {raw!r} (not found)", file=sys.stderr)
            continue

        if p.is_dir():
            for child in sorted(p.rglob("test_*.py")):
                if child.is_file():
                    pairs.append(_file_to_test_pair(child))
        elif p.is_file():
            pairs.append(_file_to_test_pair(p))
        else:
            print(f"Warning: skipping {raw!r} (not a file or directory)", file=sys.stderr)
    return pairs


# ---------------------------------------------------------------------------
# Import-mode worker (runs in a subprocess)
# ---------------------------------------------------------------------------

_IMPORT_WORKER_SCRIPT = textwrap.dedent(r"""
    import importlib.util
    import json
    import os
    import sys
    from pathlib import Path

    repo_root = Path(sys.argv[1])
    test_file = Path(sys.argv[2])
    out_path = sys.argv[3]

    if not test_file.exists():
        json.dump({"error": f"file not found: {test_file}"}, open(out_path, "w"))
        sys.exit(0)

    # Baseline modules before importing the test
    baseline = set(sys.modules.keys())

    mod_name = test_file.stem.replace("/", ".")
    try:
        spec = importlib.util.spec_from_file_location(
            f"__test_trace__.{mod_name}",
            str(test_file),
            submodule_search_locations=[],
        )
        if spec is None or spec.loader is None:
            json.dump({"error": "spec_from_file_location returned None"}, open(out_path, "w"))
            sys.exit(0)
        mod = importlib.util.module_from_spec(spec)
        # spec name is __test_trace__.X, not __main__, so guard blocks won't fire
        spec.loader.exec_module(mod)
    except SystemExit:
        pass  # Some modules call sys.exit() at import time
    except Exception as exc:
        json.dump({"error": f"{type(exc).__name__}: {exc}"}, open(out_path, "w"))
        sys.exit(0)

    new_modules = set(sys.modules.keys()) - baseline
    torch_root = str(repo_root / "torch")
    source_files = set()
    for name in new_modules:
        m = sys.modules.get(name)
        if m is None:
            continue
        f = getattr(m, "__file__", None)
        if f is None:
            continue
        f = os.path.realpath(f)
        if f.startswith(torch_root) and f.endswith(".py"):
            rel = os.path.relpath(f, str(repo_root))
            # Normalize to forward slashes
            source_files.add(rel.replace(os.sep, "/"))

    json.dump({"files": sorted(source_files)}, open(out_path, "w"))
""")


def _trace_import(
    test_name: str,
    test_file: str,
    tmpdir: str,
    timeout: int,
) -> dict[str, Any]:
    """Trace a single test via import-mode in a subprocess."""
    out_path = os.path.join(tmpdir, f"{test_name.replace('/', '_')}.json")
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_WORKER_SCRIPT, str(REPO_ROOT), test_file, out_path],
        capture_output=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )
    try:
        with open(out_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        stderr = result.stderr.decode(errors="replace")[-500:]
        return {"error": f"worker failed (rc={result.returncode}): {stderr}"}


# ---------------------------------------------------------------------------
# Coverage-mode worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def _trace_coverage(
    test_name: str,
    test_file: str,
    tmpdir: str,
    timeout: int,
) -> dict[str, Any]:
    """Trace a single test via coverage run in a subprocess."""
    if not os.path.exists(test_file):
        return {"error": f"file not found: {test_file}"}

    data_file = os.path.join(tmpdir, f".coverage.{test_name.replace('/', '_')}")
    result = subprocess.run(
        [
            sys.executable, "-m", "coverage", "run",
            "--parallel-mode",
            "--source=torch",
            f"--data-file={data_file}",
            test_file,
        ],
        capture_output=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )

    # coverage writes .coverage.<hostname>.<pid>.<random> files
    cov_files = glob_mod.glob(f"{data_file}.*")
    if not cov_files:
        # Maybe it wrote to the exact path
        if os.path.exists(data_file):
            cov_files = [data_file]

    if not cov_files:
        stderr = result.stderr.decode(errors="replace")[-500:]
        return {"error": f"no coverage data produced (rc={result.returncode}): {stderr}"}

    try:
        from coverage import CoverageData

        source_files = set()
        for cov_file in cov_files:
            cd = CoverageData(basename=cov_file)
            cd.read()
            for measured in cd.measured_files():
                real = os.path.realpath(measured)
                torch_root = str(REPO_ROOT / "torch")
                if real.startswith(torch_root) and real.endswith(".py"):
                    rel = os.path.relpath(real, str(REPO_ROOT))
                    source_files.add(rel.replace(os.sep, "/"))
        return {"files": sorted(source_files)}
    except Exception as exc:
        return {"error": f"coverage parse failed: {type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Parallel coordinator
# ---------------------------------------------------------------------------

def _run_trace(
    tests: list[tuple[str, str]],
    mode: str,
    workers: int,
    timeout: int,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Run tracing in parallel, return (results, errors).

    ``tests`` is a list of (test_name, absolute_file_path) pairs.
    """
    results: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    total = len(tests)

    trace_fn = _trace_import if mode == "import" else _trace_coverage

    with tempfile.TemporaryDirectory(prefix="trace_deps_") as tmpdir:
        done_count = 0
        start = time.monotonic()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(trace_fn, name, path, tmpdir, timeout): name
                for name, path in tests
            }

            for future in as_completed(futures):
                test_name = futures[future]
                done_count += 1
                try:
                    data = future.result()
                except subprocess.TimeoutExpired:
                    errors[test_name] = "timeout"
                    _print_progress(done_count, total, test_name, 0, 0, error="timeout")
                    continue
                except Exception as exc:
                    errors[test_name] = str(exc)
                    _print_progress(done_count, total, test_name, 0, 0, error=str(exc)[:60])
                    continue

                if "error" in data:
                    errors[test_name] = data["error"]
                    _print_progress(done_count, total, test_name, 0, 0, error="error")
                else:
                    files = data["files"]
                    results[test_name] = files
                    elapsed = time.monotonic() - start
                    _print_progress(done_count, total, test_name, len(files), elapsed)

    return results, errors


def _print_progress(
    done: int,
    total: int,
    test_name: str,
    num_files: int,
    elapsed: float,
    error: str | None = None,
) -> None:
    if error:
        print(f"  [{done}/{total}] {test_name} -- {error}", file=sys.stderr)
    else:
        print(
            f"  [{done}/{total}] {test_name} ({num_files} source files, {elapsed:.0f}s elapsed)",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------

def _aggregate(
    results: dict[str, list[str]],
    core_threshold: float | None,
) -> tuple[dict[str, dict[str, float]], int]:
    """Invert test->files to file->{test: score}, optionally filter core files."""
    inverted: dict[str, dict[str, float]] = {}
    for test_name, source_files in results.items():
        for sf in source_files:
            inverted.setdefault(sf, {})[test_name] = 1.0

    core_files_removed = 0
    # The filter is meaningless with few tests — need enough to distinguish
    # core files from domain-specific ones.
    MIN_TESTS_FOR_FILTER = 10
    if core_threshold is not None and len(results) >= MIN_TESTS_FOR_FILTER:
        cutoff = max(int(len(results) * core_threshold), 2)
        to_remove = [sf for sf, tests in inverted.items() if len(tests) >= cutoff]
        core_files_removed = len(to_remove)
        for sf in to_remove:
            del inverted[sf]

    return inverted, core_files_removed


def _write_output(
    mapping: dict[str, dict[str, float]],
    metadata: dict[str, Any],
    output_path: str,
) -> None:
    out: dict[str, Any] = {"__metadata__": metadata}
    # Sort keys for deterministic output
    for key in sorted(mapping):
        out[key] = mapping[key]

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote mapping to {output_path} ({len(mapping)} source files)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Query subcommand
# ---------------------------------------------------------------------------

def _run_query(
    mapping_path: str,
    files: list[str],
) -> None:
    with open(mapping_path) as f:
        mapping = json.load(f)

    scores: dict[str, float] = {}
    for changed_file in files:
        for test_name, score in mapping.get(changed_file, {}).items():
            scores[test_name] = scores.get(test_name, 0.0) + score

    if not scores:
        print(f"No affected tests found for {len(files)} changed file(s).")
        return

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    print(f"Affected tests for {len(files)} changed file(s):")
    for test_name, score in ranked:
        print(f"  {test_name} (score: {score:.2f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace dependencies between torch source files and test files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- trace --
    trace_p = sub.add_parser("trace", help="Build the source-to-test mapping")
    trace_p.add_argument(
        "--mode",
        choices=["import", "coverage"],
        default="import",
        help="Tracing mode (default: import)",
    )
    trace_p.add_argument(
        "--output", "-o",
        default="mapping.json",
        help="Output JSON path (default: mapping.json)",
    )
    trace_p.add_argument(
        "--workers", "-j",
        type=int,
        default=None,
        help="Number of parallel workers",
    )
    trace_p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-test timeout in seconds (default: 120 for import, 600 for coverage)",
    )
    trace_p.add_argument(
        "--include",
        default=None,
        help="Regex: only trace tests matching this pattern",
    )
    trace_p.add_argument(
        "--exclude",
        default=None,
        help="Regex: skip tests matching this pattern",
    )
    trace_p.add_argument(
        "--core-threshold",
        type=float,
        default=None,
        help="Remove source files imported by more than this fraction of tests "
             "(default: 0.85 for full discovery, disabled when tests are explicitly selected)",
    )
    trace_p.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable core-file filtering",
    )
    trace_p.add_argument(
        "files",
        nargs="*",
        help="Specific test files to trace (bypasses discovery when provided)",
    )

    # -- query --
    query_p = sub.add_parser("query", help="Look up affected tests for changed files")
    query_p.add_argument(
        "--mapping", "-m",
        required=True,
        help="Path to the mapping JSON",
    )
    query_p.add_argument(
        "--stdin",
        action="store_true",
        help="Read changed file paths from stdin (one per line)",
    )
    query_p.add_argument(
        "files",
        nargs="*",
        help="Changed source file paths (relative to repo root)",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "trace":
        cpu_count = os.cpu_count() or 4
        if args.workers is None:
            if args.mode == "import":
                args.workers = max(cpu_count - 2, 1)
            else:
                # Coverage mode is heavier per-process
                args.workers = max(cpu_count // 2, 1)

        if args.timeout is None:
            args.timeout = 120 if args.mode == "import" else 600

        explicit_selection = bool(args.files or args.include)
        if args.files:
            test_pairs = _resolve_test_files(args.files)
        else:
            discovered = _discover_python_tests()
            discovered = _filter_tests(discovered, args.include, args.exclude)
            test_pairs = [
                (t, str(TEST_DIR / (t.replace("/", os.sep) + ".py")))
                for t in discovered
            ]
        total_tests = len(test_pairs)

        print(
            f"Tracing {total_tests} tests (mode={args.mode}, workers={args.workers}, "
            f"timeout={args.timeout}s)",
            file=sys.stderr,
        )

        results, errors = _run_trace(test_pairs, args.mode, args.workers, args.timeout)

        if args.no_filter:
            core_threshold = None
        elif args.core_threshold is not None:
            core_threshold = args.core_threshold
        elif explicit_selection:
            # User narrowed the test set — filtering would discard useful data
            core_threshold = None
        else:
            core_threshold = 0.85
        mapping, core_files_removed = _aggregate(results, core_threshold)

        metadata = {
            "mode": args.mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_tests": total_tests,
            "traced_tests": len(results),
            "errors": len(errors),
            "core_threshold": core_threshold,
            "core_files_removed": core_files_removed,
        }

        if errors:
            print(f"\n{len(errors)} test(s) had errors:", file=sys.stderr)
            for test_name, err in sorted(errors.items()):
                print(f"  {test_name}: {err[:120]}", file=sys.stderr)

        _write_output(mapping, metadata, args.output)

    elif args.command == "query":
        files = list(args.files) if args.files else []
        if args.stdin:
            files.extend(line.strip() for line in sys.stdin if line.strip())
        if not files:
            parser.error("No files specified. Pass file paths as arguments or use --stdin.")
        _run_query(args.mapping, files)


if __name__ == "__main__":
    main()
