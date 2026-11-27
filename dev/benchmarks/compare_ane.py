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


def rows(document: dict) -> dict[tuple[str, int], float]:
    return {(m["scenario"], m["prompt_tokens"]): m["prefill_gpu_ms"] for m in document["measurements"]}


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

    reference = rows(alone_first)
    drift = rows(alone_last)
    candidate = rows(with_split)
    measured = [reference[key] for key in reference if key in drift]
    drift_note = ""
    if measured:
        drift_note = f"; the GPU-alone runs, {min(measured):.0f}..{max(measured):.0f} ms"

    identity = with_split.get("identity", {})
    try:
        model_name = json.loads((args.model_root / "model.json").read_text()).get("model", args.model_root.name)
    except (OSError, ValueError):
        model_name = args.model_root.name
    print(
        f"\n{identity.get('device', 'this Mac')}, {model_name}"
        f" — scenario {args.scenario} ({args.samples} sample(s) a run)"
    )
    print(
        f"split {args.experts} experts x {args.capacity} rows: "
        f"{with_split['ane_ffn_bytes'] // (1 << 20)} MiB planned, "
        f"context {alone_first['max_context_tokens']} -> {with_split['max_context_tokens']} tokens{drift_note}"
    )
    print(
        f"\n  {'scenario':20s} {'tokens':>7s} {'alone ms':>9s} {'split ms':>9s} {'speedup':>8s} "
        f"{'ttft alone':>11s} {'ttft split':>11s}"
    )
    for document in (alone_first,):
        ttft = {(m["scenario"], m["prompt_tokens"]): m["ttft_ms"] for m in document["measurements"]}
    ttft_split = {(m["scenario"], m["prompt_tokens"]): m["ttft_ms"] for m in with_split["measurements"]}
    for key in sorted(reference):
        if key not in candidate:
            continue
        speedup = reference[key] / candidate[key] if candidate[key] else 0.0
        print(
            f"  {key[0]:20s} {key[1]:7d} {reference[key]:9.0f} {candidate[key]:9.0f} {speedup:7.3f}x "
            f"{ttft.get(key, 0.0):11.0f} {ttft_split.get(key, 0.0):11.0f}"
        )
    alone_decode = [round(v) for v in alone_first["warmup"]["decode_gpu_ms"]]
    split_decode = [round(v) for v in with_split["warmup"]["decode_gpu_ms"]]
    print(f"\n  decode warmup (GPU ms, 1..{len(alone_decode)} lanes): {alone_decode} alone, {split_decode} with the split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
