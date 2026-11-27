# Performance

[Back to Splash](../README.md#performance)

## Splash 1.0 launch benchmarks

Measured for the Splash 1.0 release (September 2026) on an M5 Pro (16-core
GPU, 48 GB), serving the Qwen3.8-27B and Qwen3.6-35B-A3B Splash packages:
selected SPEED-Bench coding prompts over HTTP, a 1,024-token output limit,
reasoning on (medium for the 27B). The ratio in each cell is against the
next-fastest engine we measured. At the time, the MLX 4-bit models prepared to
the packages' target weights, byte for byte but for the 27B's 48 per-layer GDN
decay vectors, each within a float ULP, and decoded within 0.5% of them on this
M5 Pro ([upstream loading](../dev/benchmarks/upstream-loading.md)). MLX targets
now load into the weight layout of GGUF targets and run on the same kernels,
which [GGUF against llama.cpp](#gguf-against-llamacpp) measures.

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Decode · short prompt | 210 tok/s (1.7×) | 74 tok/s (2.0×) |
| Prefill · 32K prompt | 2,011 tok/s (1.3×) | 363 tok/s (1.2×) |
| Cached time to first token · 32K replay | 123 ms (6.6×) | 282 ms (7.3×) |
| Aggregate decode · 4 concurrent short prompts | 357 tok/s (2.0×) | 170 tok/s (3.9×) |

Splash 1.0 led on every measure at every prompt length we tested, and the lead
grew with load: 3.8× at four concurrent 32K requests on the 35B. The
[launch post](https://inco.ai/blog/splash/) has the method and the full
comparison against oMLX, Lily, uzu, and Ollama.

For repeatable measurements on your Mac, see [local benchmarks](validate.md#local-benchmarks).

## GGUF against llama.cpp

We compared Splash with llama.cpp (e6ab7c1, Metal) on the same Unsloth
UD-Q4_K_M files. For accuracy, both read the same text, 16,384 positions of
prose, code and chat, and at each position we compared the tokens they rank
first:

| Same token ranked first | Qwen3.8-27B | Qwen3.6-35B-A3B |
| --- | ---: | ---: |
| Splash and llama.cpp | 99.30–99.45% | 97.83–98.14% |
| llama.cpp on the CPU and on Metal | 97.8% | 96.5–96.9% |
| llama.cpp one token at a time and batched | 99.65–99.75% | 97.95% |

The positions where they differ are near-ties: there, llama.cpp's two best
tokens are a median 0.03–0.10 nats apart, against 2.6–2.7 nats over all
positions. Splash's perplexity is 0.1–0.4% (27B) and 0.1–0.9% (35B) above
llama.cpp's; llama.cpp's CPU backend is 1.7–1.8% above its Metal on the 27B.
Splash's figures cover an M5 Pro and an M3 Max with `--kv-format bf16` and
`--disable-ane`; the INT8 cache with bf16 queries (`--activations bf16`)
gives 99.23–99.25% and 97.92–97.94% on the M5 Pro.

Speed uses the selected SPEED-Bench coding prompts described above, with
greedy sampling. llama-server runs with its default settings, which do not
speculate, and for the 27B also
with the MTP draft Unsloth ships:

| Decode tok/s | M5 Pro, 20-core GPU | M3 Max, 40-core GPU |
| --- | ---: | ---: |
| Qwen3.6-35B-A3B · Splash | 175 | 209 |
| Qwen3.6-35B-A3B · llama.cpp | 69 | 66 |
| Qwen3.8-27B · Splash | 74 | 92 |
| Qwen3.8-27B · llama.cpp | 16 | 17 |
| Qwen3.8-27B · llama.cpp with MTP | 27 | 20 |

Splash decodes 2.5–3.2× as fast as llama.cpp on the 35B and 4.5–5.3× on
the 27B (2.7–4.6× against its MTP); the ratios are of the unrounded rates.
Prefilling a 2,048-token chunk, it runs at 559 tok/s against 374 on the 27B
and 3,662 against 1,968 on the 35B on the M5 Pro, and at 245 against 193 and
1,814 against 1,575 on the M3 Max.

## Smaller GGUFs on 24 GB Macs

Measured on a 24 GB M6 (12-core GPU), with each model's DFlash2 draft:

| Unsloth GGUF | Code decode | Advertised context limit |
| --- | ---: | ---: |
| `Qwen3.8-27B-GGUF:UD-IQ3_XXS` | 43.5 tok/s | 53,241 tokens (81,913 with `--language-only`) |
| `Qwen3.6-35B-A3B-GGUF:UD-Q2_K_XL` | 145 tok/s | 256K tokens |
| `Inkling-2-Mini-GGUF:UD-IQ3_XXS` | 68–79 tok/s | 160,785 tokens (197,649 with `--language-only`) |

Inkling-2-Mini's row is four 256-token answers to a code prompt over two
starts, with its vision tower loaded and its experts, 10.2 GiB, all in memory;
prose answers decoded at 53 to 59 tokens a second, and a 2,890-token prompt
prefilled in 2.6 s.

The context column reports capacity, not the prompt length of the decode
measurement. The SSD cache, on by default, sets its state staging aside within
this memory: on the M6 it costs the 27B 4,096 tokens of advertised context,
which `--max-cache-disk 0` gives back (57,337 tokens, 86,009 with
`--language-only`). Coding agents need about 100K tokens (Claude Code's own
prompt is about 33K, and at 64K it compacted repeatedly and stopped), so serve
the 27B with `--language-only` for them; with `--disable-ane` and
`--max-cache-disk 0` as well it advertises 102,393 tokens.

Splash grows into free memory only while macOS reports no memory pressure and
its spare memory, the free, file-backed and purgeable pages and half of what
compressing other applications would free, keeps a reserve of 7% of the Mac's
memory (2 to 7 GiB) and a 1 GiB margin, 3 GiB on these Macs. It gives cached
memory back when that spare memory runs short of the margin, not for macOS's
warning alone, but a request in service still takes the memory it needs within
`--max-memory`, so these models leave other applications little room. A request that cannot get memory waits for it,
and so do the requests of its priority or below that the scheduler would start
after it. While an earlier request is at work, as one prefilling a prompt near
the context limit can be for many minutes, they wait with no time limit; without
one, the request refused memory fails after 30 s with `resource_timeout`: close
memory-heavy applications, or serve with `--language-only`. To bound the wait,
give requests a `timeout`, or serve with `--request-timeout`, which bounds the
long request too. A streaming request receives a keepalive every 2 s while it
waits; a non-streaming one receives nothing until its answer, so a client's
read timeout may end it first.
Critical memory pressure can suspend a long request; the
[SSD cache](server.md#ssd-cache), on by default, keeps its progress and
cached prefixes, and writes whenever memory runs short: serving
Ternary-Bonsai-2-27B PQ2_0 to six clients' mixed traffic for 30 minutes, a
16 GiB SSD cache on the M6 wrote 26 GB, about 50 GB an hour, and read 36 GB.
`--max-cache-disk-writes` sets a write budget to spare the drive, such as
`25G` a day on a 256 GB Mac: once it is spent, states other than the ones
conversations resume from are dropped instead of written.

See [the low-bit GGUF measurements](https://github.com/incoai/splash/pull/160)
for the workloads, memory pressure, SSD settings and limitations, and
[the Bonsai measurements](https://github.com/incoai/splash/pull/166) for PQ2_0.

## The Neural Engine's expert split

A sparse MoE target can run prefill's routed experts on the Neural Engine
beside the GPU (`ops::AneMoe`). Each block keeps its most-routed experts as
int8 copies resident, and a prefill command's routes to them are marked out of
the GPU's grouping, packed into the ANE's surfaces and joined back into the
combine's rows; the GPU runs the rest of the block's routes. Decode is
untouched, its steps' rows being far below the split's least chunks. It is a
benchmark option today, not a serve option:

```bash
make benchmark-backend MODEL=unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS \
  BACKEND_ARGS="--scenario short --ane-moe-experts 8 --ane-moe-capacity 1024"
```

`--ane-moe-experts` is the experts each block keeps and `--ane-moe-capacity`
the packed rows each holds; their product is the rows the join writes, at most
8,192. Run each configuration twice and read the second run: the first after a
build learns which experts the routes reach, keeps the counts under the
build's calibration key, and only the next start plans the residents from
them. A start says what it took on its log line, and `/status` reports it
(`ane_moe`: state, copies, rows, blocks, join rows, bytes, commands,
evaluations).

Measured on an M4 Pro (16-core GPU, 24 GB), Inkling-2-Mini `UD-IQ3_XXS`, 8
experts of 512 packed rows on all 24 blocks (1.5 GiB), prefill GPU
milliseconds, medians of interleaved runs:

| Prompt tokens | GPU alone | with the split | |
| ---: | ---: | ---: | ---: |
| 480 | 979 | 858 | 1.14× |
| 640 | 1,283 | 998 | 1.29× |
| 1,024 | 1,998 | 1,588 | 1.26× |
| 1,536 | 3,003 | 2,561 | 1.17× |
| 2,016 | 3,971 | 3,528 | 1.13× |

Long prompts hold the gain: a 30,000-token prompt of this repository's
documentation (teacher-forced, its windows cold) prefilled 37.2 s alone and
32.4 s with the split at 17,400 tokens, 1.15×, and 67.8 s against 59.1 s at
29,992. Capacity sets the point: 8 experts of 1,024 rows (8,192 join rows) is the fastest from about
1,000 tokens up — 3,104 ms against 3,971 at 2,016 — and the slowest at 480
(1,206 ms against 979), because the join's scale, pack, evaluation and copy
run over every column whether the routes fill it or not. The copies also cost
context: on a 24 GB Mac, 1.5 GiB of residents leaves no host memory for a 32K
context beside them, and a larger Mac is where both fit.

The quality the split costs, measured as the KL of its next-token distribution
from the GPU's over identical ids: on 516 teacher-forced rows of that 30,000-token
prompt, mean 5.6e-3 a row, median 2.6e-3, p99 4.9e-2, with the most likely
token agreeing on 96% of them. A weight scale each group of 32 inputs halves
the requantization error of one a row and changes the end-to-end KL not at all
— the input's per-row int8 and the graph's fp16 arithmetic are what it is made
of. To measure a window on your Mac:

```bash
build/engine-tests/teacher-logits build/splash.metallib MODEL_ROOT TOKENS OUTPUT --window 17400:128
build/engine-tests/teacher-logits build/splash.metallib MODEL_ROOT TOKENS OUTPUT-ANE --window 17400:128 \
  --ane-moe-experts 8 --ane-moe-capacity 1024
.venv/bin/python -m dev.benchmarks.teacher_kl OUTPUT-17400.f32 OUTPUT-ANE-17400.f32
```

`TOKENS` is the prompt's little-endian uint32 ids from the model's
`tokenizer/tokenizer.json`; `MODEL_ROOT` is the model directory (`make -n
benchmark-backend MODEL=... | tail -1` prints it).
