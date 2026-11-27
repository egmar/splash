"""Compare two teacher-logits dumps: the KL of the candidate's next-token
distribution from the reference's, and whether their most likely tokens agree.

    python -m dev.benchmarks.teacher_kl REFERENCE.f32 CANDIDATE.f32 [--vocab N]
                                        [--max-mean-kl X] [--min-top1 X]

Each file is what `teacher-logits` writes (OUTPUT-END.f32): ROWS + 1 rows of
the target's vocabulary as little-endian fp32, the row of the prefill's last
position first. Rows are compared in order. Vocabulary entries that are not
finite in either dump are masked out (a MoE target's reserved ids are -inf).

Run one window list without the split and the same window list with it
(`--ane-moe-experts N --ane-moe-capacity ROWS`) and the reference is what the
GPU alone predicted. Exit status 1 past --max-mean-kl or below --min-top1.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def load(path: str, vocabulary: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if not vocabulary or values.size % vocabulary:
        raise SystemExit(f"{path}: {values.size} values are not rows of {vocabulary}")
    return values.reshape(values.size // vocabulary, vocabulary).astype(np.float64)


def log_softmax(values: np.ndarray, finite: np.ndarray) -> np.ndarray:
    masked = np.where(finite, values, -np.inf)
    shifted = masked - masked.max(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="the GPU alone's OUTPUT-END.f32")
    parser.add_argument("candidate", help="the split's OUTPUT-END.f32")
    parser.add_argument("--vocab", type=int, default=201024, help="the model's vocabulary size")
    parser.add_argument("--max-mean-kl", type=float, help="fail past this mean KL")
    parser.add_argument("--min-top1", type=float, help="fail below this top-1 agreement")
    args = parser.parse_args(argv)

    reference = load(args.reference, args.vocab)
    candidate = load(args.candidate, args.vocab)
    if reference.shape != candidate.shape:
        raise SystemExit(f"the dumps hold {reference.shape[0]} and {candidate.shape[0]} rows")

    finite = np.isfinite(reference) & np.isfinite(candidate)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_reference = log_softmax(reference, finite)
        log_candidate = log_softmax(candidate, finite)
        difference = np.where(finite, log_reference - log_candidate, 0.0)
        probability = np.where(finite, np.exp(log_reference), 0.0)
        kl = (probability * difference).sum(axis=1)
    top1 = np.where(finite, reference, -np.inf).argmax(axis=1) == np.where(finite, candidate, -np.inf).argmax(
        axis=1
    )
    agreement = float(top1.mean())
    print(
        f"rows {kl.size}  mean KL {kl.mean():.3e}  median {np.median(kl):.3e}  "
        f"p90 {np.quantile(kl, 0.9):.3e}  p99 {np.quantile(kl, 0.99):.3e}  max {kl.max():.3e}  "
        f"top-1 {int(top1.sum())}/{top1.size} ({agreement:.1%})  rows over 1e-2 {int((kl > 1e-2).sum())}"
    )
    failed = False
    if args.max_mean_kl is not None and kl.mean() > args.max_mean_kl:
        print(f"mean KL {kl.mean():.3e} is past {args.max_mean_kl:.3e}", file=sys.stderr)
        failed = True
    if args.min_top1 is not None and agreement < args.min_top1:
        print(f"top-1 {agreement:.1%} is below {args.min_top1:.1%}", file=sys.stderr)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
