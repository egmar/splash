import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from dev.benchmarks import compare_ane


def document(ane_bytes: int, context: int, prefill: tuple[int, int], decode: tuple[int, ...]) -> str:
    """What backend-benchmark prints, one line, as its reader takes it."""
    return json.dumps({
        "schema_version": 2,
        "identity": {"device": "this Mac"},
        "ane_ffn_bytes": ane_bytes,
        "max_context_tokens": context,
        "measurements": [
            {"scenario": "short", "prompt_tokens": tokens, "prefill_gpu_ms": milliseconds,
             "ttft_ms": milliseconds + 10, "cache_status": "miss"}
            for tokens, milliseconds in zip((480, 2016), prefill)
        ],
        "warmup": {"decode_gpu_ms": list(decode)},
    }, separators=(",", ":"))


def canned(split_bytes: int) -> str:
    """A benchmark that prints those documents, scaled when the split is asked
    for, so the comparison's table can be read without a Mac's GPU:
    `split_bytes` of zero says the split was not taken."""
    split = document(split_bytes, 91153, (1530, 3210), (59, 92, 121, 152))
    alone = document(0, 197649, (975, 3916), (58, 88, 116, 144))
    return f"""#!/bin/sh
for argument in "$@"; do
  if [ "$argument" = "--ane-moe-experts" ]; then
    echo '{split}'
    exit 0
  fi
done
echo '{alone}'
"""


class CompareAneTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.benchmark = self.benchmark_of("benchmark", 1686900000)

    def benchmark_of(self, name: str, split_bytes: int) -> Path:
        path = self.root / name
        path.write_text(canned(split_bytes))
        os.chmod(path, 0o755)
        return path

    def compare(self, benchmark: Path | None = None, *arguments: str) -> str:
        """The comparison's stdout over the canned benchmark."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            result = compare_ane.main([
                str(self.root / "splash.metallib"), str(self.root / "model"),
                "--scenario", "short", "--benchmark", str(benchmark or self.benchmark), *arguments,
            ])
        self.assertEqual(result, 0)
        return output.getvalue()

    def test_the_table_reads_alone_fresh_and_calibrated(self):
        output = self.compare()
        self.assertIn("fresh", output)
        self.assertIn("calibrated", output)
        # 975 ms alone, 1530 ms with the first split run's counts, the same
        # again once calibrated, and the change the split makes.
        self.assertIn("short 480", output)
        self.assertIn("975", output)
        self.assertIn("1530 (0.64x)", output)
        self.assertIn("+57%", output)
        self.assertIn("decode warmup", output)

    def test_the_json_is_printed_only_when_asked_for(self):
        self.assertNotIn('{"schema_version"', self.compare())
        printed = [line for line in self.compare(None, "--json").splitlines() if line.startswith('{"schema_version"')]
        self.assertEqual(len(printed), 1)
        self.assertEqual(json.loads(printed[0])["ane_ffn_bytes"], 1686900000)

    def test_a_split_the_start_did_not_take_is_refused(self):
        # A benchmark whose split runs took nothing says so rather than
        # comparing two GPU-alone columns.
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.compare(self.benchmark_of("no-split", 0))
        self.assertIn("took no split", str(raised.exception))

    def test_a_benchmark_that_is_not_built_is_refused(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.compare(self.root / "missing")
        self.assertIn("is not built", str(raised.exception))
