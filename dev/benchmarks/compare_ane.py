"""Run backend-benchmark twice and print what the Neural Engine's expert split
does, so one command gives the comparison:

    python -m dev.benchmarks.compare_ane METALLIB MODEL_ROOT [--scenario short,partial]
        [--context-lengths 2048,8192] [--experts 8] [--capacity 1024]

The scenarios run once without the split and twice with it, alternating the
GPU-alone runs either side so the Mac's drift shows up in the reference too;
--verbose streams each pass's log and measurements, --json prints the compared
pass's JSON, and every pass's JSON is kept in a directory the run names;
the last split run is the one compared, because the first run after a build is
uncalibrated (the split learns which experts the routes reach and plans the
next start from them). `--experts x --capacity` at most 8,192 packed rows, the
join region's most.

The split's own log line, and the benchmark's table, still go to the terminal:
only the JSON is read here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def power_warning() -> str:
    """What the Mac is drawing from, and a warning when it is throttling: a
    battery is not a comparable measure against a plugged-in run."""
    try:
        result = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    if "Battery Power" in result.stdout and "discharging" in result.stdout:
        return ("this Mac draws from its battery: the GPU is power-throttled and these numbers do not"
                " compare with a plugged-in run")
    return ""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metallib", type=Path)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--scenario", default="short", help="comma-separated: short, partial, context, decode...")
    parser.add_argument("--context-lengths", help="the context scenario's lengths, comma-separated")
    parser.add_argument("--experts", type=int, default=8, help="experts each block keeps (default 8)")
    parser.add_argument("--capacity", type=int, default=1024, help="packed rows each holds (default 1024)")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--benchmark", type=Path, default=Path("build/engine-tests/backend-benchmark"))
    parser.add_argument("--json-dir", type=Path, help="where each pass's JSON is kept (default: a temporary directory)")
    parser.add_argument("--json", action="store_true", help="also print the compared passes' JSON on stdout")
    parser.add_argument("--verbose", action="store_true", help="stream each pass's log and measurements")
    return parser.parse_args(argv)


def run(benchmark: Path, metallib: Path, model_root: Path, args: list[str], label: str,
        verbose: bool = False) -> dict:
    """One benchmark run's JSON, its measurements and log passing through as
    they come."""
    print(f"\n[{label}]", file=sys.stderr)
    result = subprocess.run(
        [str(benchmark), str(metallib), str(model_root), *args],
        stdout=subprocess.PIPE,
        stderr=None if verbose else subprocess.PIPE,
        text=True,
    )
    document = None
    for line in result.stdout.splitlines():
        if line.startswith('{"schema_version"'):
            document = line
    if result.returncode != 0 or document is None:
        sys.stderr.write(result.stdout[-2000:])
        if result.stderr:
            sys.stderr.write(result.stderr[-2000:])
        raise SystemExit(f"the benchmark failed (exit {result.returncode})")
    return json.loads(document)


# The context scenario's forms, which its rows are named by.
CONTEXT_FORMS = {"cold", "exact", "rolling", "rolling_cold", "answer", "continuation", "continuation_cold",
                 "lazy_seed", "lazy_chain_end", "lazy_repeat", "lazy_materialize", "lazy_reuse"}


def label(scenario: str, tokens: int) -> str:
    """A row's test: what it is, and the prompt it prefilled."""
    if scenario in CONTEXT_FORMS:
        return f"context {scenario.replace('_', ' ')} {tokens}"
    if scenario.startswith("partial_4k_"):
        return f"partial {scenario[len('partial_4k_'):]} {tokens}"
    return f"{scenario} {tokens}"


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.experts * args.capacity > 8192:
        raise SystemExit("--experts x --capacity is at most 8,192 packed rows")
    if not args.benchmark.exists():
        raise SystemExit(f"{args.benchmark} is not built: make build/engine-tests/backend-benchmark")
    common = ["--json", "--scenario", args.scenario, "--samples", str(args.samples)]
    if args.context_lengths:
        common += ["--context-lengths", args.context_lengths]
    split = common + ["--ane-moe-experts", str(args.experts), "--ane-moe-capacity", str(args.capacity)]

    power = power_warning()
    if power:
        print(f"warning: {power}\n", file=sys.stderr)
    print("alone, with the split twice, then alone again; the second split run is compared", file=sys.stderr)
    alone_first = run(args.benchmark, args.metallib, args.model_root, common, "1/4 alone", args.verbose)
    calibrating = run(args.benchmark, args.metallib, args.model_root, split, "2/4 split (calibrating)", args.verbose)
    with_split = run(args.benchmark, args.metallib, args.model_root, split, "3/4 split (compared)", args.verbose)
    alone_last = run(args.benchmark, args.metallib, args.model_root, common, "4/4 alone", args.verbose)
    for document, description in ((calibrating, "the first split run"), (with_split, "the split")):
        if document["ane_ffn_bytes"] == 0:
            raise SystemExit(f"{description} took no split: see its log for why")

    def measured(document: dict) -> dict[tuple[str, int], dict[str, float]]:
        return {(m["scenario"], m["prompt_tokens"]): m for m in document["measurements"]}

    alone = measured(alone_first)
    alone_last_rows = measured(alone_last)
    split_rows = measured(with_split)
    if args.json_dir:
        args.json_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.json_dir = Path(tempfile.mkdtemp(prefix="ane-compare-"))
    kept = {}
    for name, document in (("alone-1", alone_first), ("split-calibrating", calibrating),
                           ("split", with_split), ("alone-2", alone_last)):
        path = args.json_dir / f"{name}.json"
        path.write_text(json.dumps(document, indent=2) + "\n")
        kept[name] = path
    if args.json:
        print(json.dumps(with_split))

    identity = with_split.get("identity", {})
    try:
        model_name = json.loads((args.model_root / "model.json").read_text()).get("model", args.model_root.name)
    except (OSError, ValueError):
        model_name = args.model_root.name
    drifts = [(row_key, alone[row_key]["prefill_gpu_ms"], alone_last_rows[row_key]["prefill_gpu_ms"])
              for row_key in alone if row_key in alone_last_rows]
    worst = max(drifts, key=lambda row: abs(row[2] - row[1]) / row[1] if row[1] else 0.0, default=None)
    print(f"\n{identity.get('device', 'this Mac')}, {model_name} — {args.scenario}")
    print(
        f"split {args.experts} experts x {args.capacity} rows, {with_split['ane_ffn_bytes'] // (1 << 20)} MiB:"
        f" context {alone_first['max_context_tokens']} alone, {with_split['max_context_tokens']} with it"
    )

    if worst and worst[1]:
        print(
            f"the GPU alone, both passes: {worst[1]:.0f} and {worst[2]:.0f} ms at {label(worst[0][0], worst[0][1])}"
            f" ({(worst[2] - worst[1]) / worst[1] * 100.0:+.0f}%)"
        )
        if abs(worst[2] - worst[1]) / worst[1] > 0.10:
            print("warning: the two GPU-alone passes disagree by more than a tenth: the Mac is drifting (power or heat)"
                  " and these numbers will not repeat — run again plugged in and cool")
        if power:
            print(f"warning: {power}")

    def median(row_key: tuple[str, int], key: str) -> float:
        values = [row[key] for row in (alone.get(row_key), alone_last_rows.get(row_key)) if row]
        return statistics.median(values) if values else 0.0

    def measure(document: dict, row_key: tuple[str, int], key: str) -> float:
        row = measured(document).get(row_key)
        return row[key] if row else 0.0

    def rate(row_key: tuple[str, int], milliseconds: float) -> float:
        return 1000.0 * row_key[1] / milliseconds if milliseconds else 0.0

    def change(before: float, after: float) -> str:
        if not before:
            return "-"
        return f"{(after - before) / before * 100.0:+.0f}%"

    def speedup(before: float, after: float) -> str:
        return f"{before / after:.2f}x" if after else "-"

    def table(title: str, value, shown: str) -> None:
        """A measure's table: alone, the first split run's fresh counts, the
        calibrated run's, and the change the second makes on the first."""
        print(
            f"\n  {title}\n  {'test':28s} {'alone':>10s} {'fresh':>16s} {'calibrated':>16s} {'delta':>8s}"
        )
        for row_key in sorted(alone):
            if row_key not in split_rows or row_key not in measured(calibrating):
                continue
            before, fresh, after = value(row_key)
            print(
                f"  {label(row_key[0], row_key[1]):28s} {shown(before):>10s} "
                f"{shown(fresh) + ' (' + speedup(before, fresh) + ')':>16s} "
                f"{shown(after) + ' (' + speedup(before, after) + ')':>16s} {change(before, after):>8s}"
            )

    table(
        "prefill GPU ms (fresh: the first split run's counts, which it calibrates for the next)",
        lambda row_key: (median(row_key, "prefill_gpu_ms"),
                         measure(calibrating, row_key, "prefill_gpu_ms"),
                         measure(with_split, row_key, "prefill_gpu_ms")),
        lambda value: f"{value:.0f}",
    )
    if args.verbose:
        table(
            "prefill tokens a second",
            lambda row_key: (rate(row_key, median(row_key, "prefill_gpu_ms")),
                             rate(row_key, measure(calibrating, row_key, "prefill_gpu_ms")),
                             rate(row_key, measure(with_split, row_key, "prefill_gpu_ms"))),
            lambda value: f"{value:.0f}",
        )
        table(
            "time to first token, ms",
            lambda row_key: (median(row_key, "ttft_ms"), measure(calibrating, row_key, "ttft_ms"),
                             measure(with_split, row_key, "ttft_ms")),
            lambda value: f"{value:.0f}",
        )
    alone_decode = [round(v) for v in alone_first["warmup"]["decode_gpu_ms"]]
    split_decode = [round(v) for v in with_split["warmup"]["decode_gpu_ms"]]
    print(
        f"\n  decode warmup (GPU ms, 1..{len(alone_decode)} lanes): {alone_decode} alone, {split_decode} with the"
        f" split ({', '.join(change(before, after) for before, after in zip(alone_decode, split_decode))})"
    )
    print("\n  JSON of each pass: " + ", ".join(str(path) for path in kept.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
