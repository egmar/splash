"""Run backend-benchmark twice and print what the Neural Engine's expert split
does, so one command gives the comparison:

    python -m dev.benchmarks.compare_ane METALLIB MODEL_ROOT [--scenario short,partial]
        [--context-lengths 2048,8192] [--experts 8] [--capacity 1024]

The scenarios run once without the split and twice with it, alternating the
GPU-alone runs either side so the Mac's drift shows up in the reference too;
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
    return parser.parse_args(argv)


def run(benchmark: Path, metallib: Path, model_root: Path, args: list[str], label: str) -> dict:
    """One benchmark run's JSON, its measurements and log passing through as
    they come."""
    print(f"\n[{label}]", file=sys.stderr)
    result = subprocess.run(
        [str(benchmark), str(metallib), str(model_root), *args],
        stdout=subprocess.PIPE,
        text=True,
    )
    document = None
    for line in result.stdout.splitlines():
        if line.startswith('{"schema_version"'):
            document = line
    if result.returncode != 0 or document is None:
        sys.stderr.write(result.stdout[-2000:])
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
    common = ["--scenario", args.scenario, "--samples", str(args.samples)]
    if args.context_lengths:
        common += ["--context-lengths", args.context_lengths]
    split = common + ["--ane-moe-experts", str(args.experts), "--ane-moe-capacity", str(args.capacity)]

    print("alone, with the split twice, then alone again; the second split run is compared", file=sys.stderr)
    alone_first = run(args.benchmark, args.metallib, args.model_root, common, "1/4 alone")
    calibrating = run(args.benchmark, args.metallib, args.model_root, split, "2/4 split (calibrating)")
    with_split = run(args.benchmark, args.metallib, args.model_root, split, "3/4 split (compared)")
    alone_last = run(args.benchmark, args.metallib, args.model_root, common, "4/4 alone")
    for document, label in ((calibrating, "the first split run"), (with_split, "the split")):
        if document["ane_ffn_bytes"] == 0:
            raise SystemExit(f"{label} took no split: see its log for why")

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
    print(f"\n{identity.get('device', 'this Mac')}, {model_name} — {args.scenario}")
    print(
        f"split {args.experts} experts x {args.capacity} rows: {with_split['ane_ffn_bytes'] // (1 << 20)} MiB planned;"
        f" context {alone_first['max_context_tokens']} alone, {with_split['max_context_tokens']} with the split"
    )

    def median(row_key: tuple[str, int], key: str) -> float:
        values = [row[key] for row in (alone.get(row_key), alone_last_rows.get(row_key)) if row]
        return statistics.median(values) if values else 0.0

    def change(before: float, after: float) -> str:
        if not before:
            return "-"
        return f"{(after - before) / before * 100.0:+.0f}%"

    print(
        f"\n  {'test':28s} {'prefill ms':>26s} {'tokens/s':>26s} {'ttft ms':>26s}"
    )
    print(
        f"  {'':28s} {'alone':>8s} {'split':>8s} {'delta':>8s}"
        f" {'alone':>8s} {'split':>8s} {'delta':>8s}"
        f" {'alone':>8s} {'split':>8s} {'delta':>8s}"
    )
    for row_key in sorted(alone):
        candidate_row = split_rows.get(row_key)
        if not candidate_row:
            continue
        alone_ms, split_ms = median(row_key, "prefill_gpu_ms"), candidate_row["prefill_gpu_ms"]
        alone_rate = 1000.0 * row_key[1] / alone_ms if alone_ms else 0.0
        split_rate = 1000.0 * row_key[1] / split_ms if split_ms else 0.0
        alone_ttft, split_ttft = median(row_key, "ttft_ms"), candidate_row["ttft_ms"]
        change_ms = change(alone_ms, split_ms)
        change_rate = change(alone_rate, split_rate)
        change_ttft = change(alone_ttft, split_ttft)
        print(
            f"  {label(row_key[0], row_key[1]):28s} {alone_ms:8.0f} {split_ms:8.0f} {change_ms:>8s}"
            f" {alone_rate:8.0f} {split_rate:8.0f} {change_rate:>8s}"
            f" {alone_ttft:8.0f} {split_ttft:8.0f} {change_ttft:>8s}"
        )
    alone_decode = [round(v) for v in alone_first["warmup"]["decode_gpu_ms"]]
    split_decode = [round(v) for v in with_split["warmup"]["decode_gpu_ms"]]
    change_decode = [
        f"{change(before, after) if before else '-'}" for before, after in zip(alone_decode, split_decode)
    ]
    print(
        f"\n  decode warmup (GPU ms, 1..{len(alone_decode)} lanes): {alone_decode} alone, {split_decode} with the"
        f" split ({', '.join(change_decode)})"
    )
    print("\n  JSON of each pass: " + ", ".join(str(path) for path in kept.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
