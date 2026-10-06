# Model families in System One Studio

Every System One model reads a state and answers typed questions (choice, score, noul) in one
pass, but the published models are built in seven different ways. A **family** is one of those
ways (`systemone_studio/families.py`). Whether one model trains here depends on two things:

- the studio has a trainer for its checkpoint format, its **kind** (`systemone_studio/kinds.py`);
- its licence allows derivatives (`families.LICENCE_POLICY`).

The studio never hides a model. One that cannot train here, or will not fit this machine, is
listed with the reason and what to do instead.

## Licence policy

| Licence kind | Train a fine-tune | Publish a fine-tune |
|---|---|---|
| open (Apache-2.0, MIT) | yes | yes, under the same licence |
| share-alike (CC-BY-SA-4.0) | yes | yes, under the same licence |
| non-commercial (CC-BY-NC-4.0) | yes, for research or personal use | **no**: systemonemodels.tech also sells hosted inference, so publishing there is not clearly non-commercial use (a decision for the owner, see below) |
| none (no licence published) | **no**: all rights reserved, no derivatives | no |
| closed (no weights) | no | no |

Every licence below was re-read from its Hugging Face record on 2026-10-04 and matches the
catalogue. A fine-tune's own base can have a base too (Decider on Qwen3.5-Base, Julia 1 on
mmBERT-small, Nimble on Qwen3.5-9B): all of those are permissive; Jev-Omni's Gemma 4 base and
CLM's Qwen3-8B base should be read again when their trainers come.

## The kinds the studio trains today

| Kind | Models | Trainer | Prompt and readout | Checkpoint written | Exports | NoulXP |
|---|---|---|---|---|---|---|
| `laya` | convaiinnovations/laya, aac6fef/laya-mlx, aac6fef/laya-multilingual-mlx, aac6fef/laya-typed-decisions-mlx | `engine.py` (MLX), `torch_engine.py` (PyTorch): LoRA (DoRA, rsLoRA, LoRA+), head only, or full top layers; proper / RLCD / CE objectives | `[CLS] <type> question: … [SEP] [MASK] name: description … [SEP] state [SEP]`, scorer at each marker | Laya's (FP16 safetensors, `rl_agent_config.json` with refitted temperatures) | ONNX, Core ML (float, int8, int4), NoulXP | `encoder-markers` via `noulxp export laya`, conformance from the `laya` package |
| `julia` | SupersonicLabs/Julia-1 | the same two trainers, with Julia's prompt and files (`julia.py`); LoRA, head only, or a full fine-tune of all 22 layers | Julia's: an option is its description (its name without one), 8,192 tokens, 512 for the head, 48 per option | Julia's (`julia_config.json`, `inference-policy.json`, float32 weights: LoRA merged into the base's own float32 tensors, the fitted temperature folded into the scorer) | NoulXP | `encoder-markers` via `noulxp export julia`, conformance from Julia's own inference (`noulxp.native.julia`) |
| `decider` | Mapika/decider-0.8b, decider-2b, decider-4b | `decider_engine.py` (PyTorch + PEFT; 4-bit QLoRA with bitsandbytes on CUDA), `decider_mlx.py` (MLX-LM; 4-bit with MLX): LoRA on the attention and MLP projections, as Decider v11 was made; a proper scoring rule on the option letters at the answer slot (measured better than Decider's own cross-entropy, which stays an option) | Decider's plain state-first layout, one question per row, isolated score levels (one yes/no row per level), at most 10 options per training row with the gold kept | Decider's (merged bfloat16 safetensors, its tokenizer and `decider/` code, `decider_config.json` with temperatures fitted per answer type) | merged safetensors, GGUF bf16 / q8_0 / f16 (llama.cpp converter at the commit llama-cpp-python 0.3.35 vendors, 4df29be), MLX-LM int4 / int8, NoulXP | `causal-letters` via `noulxp export decider` over the run's bf16 GGUF (q8_0 on request), conformance from Decider's own GGUF readout (`noulxp.native.decider`) |

No change to the noulxp library was needed: its 0.4 release already exports Julia 1 and Decider
checkpoints and ships both makers' inference as native runtimes.

## Every family

`Train` and `Publish` say what the licence allows (the policy above); `Trainer here` says whether
the studio has the trainer for that model's format today.

### Encoder + option-marker head (`laya`) — trainer: partial, NoulXP `encoder-markers`

| Model | Licence | Train | Publish | Trainer here | Exports | Status |
|---|---|---|---|---|---|---|
| convaiinnovations/laya | Apache-2.0 | yes | yes | laya | ONNX, Core ML, NoulXP | trains here (shipped) |
| aac6fef/laya-mlx | Apache-2.0 | yes | yes | laya | ONNX, Core ML, NoulXP | trains here (shipped) |
| aac6fef/laya-multilingual-mlx | Apache-2.0 | yes | yes | laya | ONNX, Core ML, NoulXP | trains here (shipped) |
| aac6fef/laya-typed-decisions-mlx | Apache-2.0 | yes | yes | laya | ONNX, Core ML, NoulXP | trains here (shipped) |
| SupersonicLabs/Julia-1 | Apache-2.0 | yes | yes | julia | NoulXP | **trains here (new)** |
| wfzyx/von | Apache-2.0 | yes | yes | – | – | next: von-sdk's trainer, order-invariant option mask |
| com-kotobalabs/open-jev-deberta-v3-large | Apache-2.0 | yes | yes | – | – | next: DeBERTa-v3 marker head |
| mpnikhil/dev-0.4b | Apache-2.0 | yes | yes | – | – | next: own format |
| alibiserikbay/JevK5-Lite | Apache-2.0 | yes | yes | – | – | next: DeBERTa-v3 label-marker head |

### Decoder, letter readout (`letter`) — trainer: partial, NoulXP `causal-letters`

| Model | Licence | Train | Publish | Trainer here | Exports | Status |
|---|---|---|---|---|---|---|
| Mapika/decider-2b | Apache-2.0 | yes | yes | decider | safetensors, GGUF, MLX-LM, NoulXP | **trains here (new)** |
| Mapika/decider-0.8b | Apache-2.0 | yes | yes | decider | safetensors, GGUF, MLX-LM, NoulXP | **trains here (new)** |
| Mapika/decider-4b | Apache-2.0 | yes | yes | decider | safetensors, GGUF, MLX-LM, NoulXP | **trains here (new)** |
| alibiserikbay/JevK5, JevK5-2B, JevK5-9B | Apache-2.0 | yes | yes | – | – | next: own prompt, knockout rounds above 16 options |
| bespokelabs/Bespoke-Nimble-9B | Apache-2.0 | yes | yes | – | – | next: LoRA adapter on Qwen3.5-9B, codes A–Z, AA… |
| cua-ai/cua-s1-4b-0.2 | Apache-2.0 | yes | yes | – | – | next: LoRA on Qwen3.5-4B |
| chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2 | Apache-2.0 | yes | yes | – | – | next |
| caiovicentino1/Eikos-4B | MIT | yes | yes | – | – | next |
| openjev/openjev (27B) | CC-BY-NC-4.0 | research only | no | – | – | later (27B, non-commercial) |
| togethercomputer/Tev1-4B, Tev1-0.8B | none | no | no | – | – | not trainable: no licence |
| typesafe/jev | proprietary, no weights | no | no | – | – | API only |

### Per-option cross-encoder (`crossenc`) — trainer: next, NoulXP `encoder-pairs` (unreleased)

| Model | Licence | Train | Publish | Trainer here | Status |
|---|---|---|---|---|---|
| AlexWortega/openjev | MIT | yes | yes | – | next |
| mobarmg/jev-schema-scorer-deberta-v3-large | MIT | yes | yes | – | next |
| argos1111/modernbert-ja-310m-jev | CC-BY-SA-4.0 | yes | yes, CC-BY-SA | – | next |
| pngwn/system-one-qwen3.5-4b-scorer | CC-BY-NC-4.0 | research only | no | – | next, private use only |

### Decoder + learned pointer or slot head (`head`) — trainer: planned, no NoulXP profile

| Model | Licence | Train | Publish | Status |
|---|---|---|---|---|
| jaredpalmer/kev-4b, kev-0.6b | Apache-2.0 | yes | yes | planned: Kev's own trainer |
| iapp/OpenThai-SystemOne | Apache-2.0 | yes | yes | planned: 256-slot head, control tokens |
| akhilaaa3/Jev-Omni (12B, multimodal) | Apache-2.0 | yes | yes | planned: no published trainer |
| C-Tianyu/NanoJev | none | no | no | not trainable: no licence |

### GLiNER2 / GLiClass (`gliner`) — trainer: planned, no NoulXP profile

| Model | Licence | Train | Publish | Status |
|---|---|---|---|---|
| fastino/GLiNER2.5-Decide, GLiNER2.5-Decide-1B | Apache-2.0 | yes | yes | planned: gliner2's own trainer |
| heman10x/rlcd-modernbert-151m | Apache-2.0 | yes | yes | planned: GLiClass trainer |

### Frozen embedder + heads (`embedder`) and tiny scorers (`tiny`) — planned, no NoulXP profile

| Model | Licence | Train | Publish | Status |
|---|---|---|---|---|
| Contrastive-LM/CLM-v0.1-8B | Apache-2.0 | yes | yes | planned: heads on frozen Qwen3-8B embeddings |
| cua-ai/cua-s1-forms | MIT | yes | yes | planned: byte-level scorer from scratch, CPU |
| cua-ai/cua-s1-nano-0.1 | Apache-2.0 | yes | yes | planned |

SAGEA's Mira is deliberately not in the catalogue (exclusive launch, benchmark pending).

## Measured on a GPU (2026-10-04)

One RunPod A40 (48 GB, driver 580, PyTorch 2.11 with CUDA 12.8, transformers 5.17,
flash-linear-attention 0.5.2, PEFT 0.21, llama-cpp-python 0.3.35, noulxp 0.4.0), driven through
the studio's own job runner exactly as the UI starts jobs. Dataset: the studio's built-in
**Emotion** example (dair-ai/emotion, six labels, one choice question): 1,079 training rows, 121
validation rows, 600 held-out test rows that no run trains or selects on. Every number below is
from the runs' own reports; the raw logs are in the research workspace
(`results/studio-families/`).

### Before and after fine-tuning (600 test decisions)

| Run | Recipe | Accuracy before → after | Macro-F1 | McNemar (fixed / broken, p) | ECE | Log loss | Training | Peak GPU memory |
|---|---|---|---|---|---|---|---|---|
| Julia 1 | LoRA r16 on all 22 encoder layers + head, 4 epochs, frozen weights in float32 | 82.5% → **92.2%** (95% CI 89.7–94.1) | 0.821 → 0.920 | 78 / 20, p = 2.9e-9 | 0.102 → **0.021** | 0.696 → 0.249 | 46 s | 1.6 GB |
| Julia 1 | full fine-tune of all 22 layers + head, lr 3e-5, 4 epochs | 82.5% → **92.7%** (90.3–94.5) | 0.821 → 0.925 | 81 / 20, p = 6.9e-10 | 0.102 → 0.033 | 0.696 → 0.251 | 35 s | 2.0 GB |
| Decider 2B | LoRA r16 (alpha 32) on the attention and MLP projections, cross-entropy on the letters, 2 epochs, lr 1e-4 | 82.5% → **89.2%** (86.4–91.4) | 0.821 → 0.890 | 74 / 34, p = 1.5e-4 | 0.021 → 0.053 | 0.499 → 0.480 | 151 s | 4.2 GB |

Both base models happen to score 495 of 600 (82.5%), with different confusion matrices. Julia's
fitted temperature, folded into its scorer, was 1.57 (LoRA) and 1.99 (full): the fine-tunes
were overconfident, and folding the temperature in is what brings their ECE down. Decider's
choice temperature was refitted from 1.164 to 1.096 on 121 validation decisions; on the test
split this cross-entropy fine-tune is more confident than it is right (ECE 0.053). The proper
scoring rule, below, does better (0.029) and is now the default; a larger validation split
would calibrate either one better. Latency is unchanged by
fine-tuning (paired, one question per request on the A40): Julia 10.4 ms, Decider 36.8 ms.

### Exports and NoulXP packages

| Model | Export | Size | Checked against | Result |
|---|---|---|---|---|
| Julia 1 (LoRA and full) | NoulXP `encoder-markers` (ONNX graph over the run's own weights) | 586 MB | Julia's own inference on the CPU, NoulXP's 52 requests | **52/52 cases**, 89/89 same answers, the 2 requests Julia refuses refused alike, max \|Δp\| 5.0e-5 |
| Decider 2B | GGUF bf16 | 3.6 GB | merged weights in float32, 60 test rows | 60/60 same answers, max \|Δp\| 0.009 |
| Decider 2B | GGUF q8_0 | 1.9 GB | merged weights in float32, 60 test rows | 60/60 same answers, max \|Δp\| 0.022 |
| Decider 2B (the maker's v11) | GGUF q8_0 | 2.0 GB | float32, the typed-decisions test split (2,000 questions), on an A40 | **21 answers changed** (16 beyond ties), max \|Δp\| 0.049 |
| Decider 2B (the maker's v11) | GGUF bf16 | 3.8 GB | the same | 0 answers changed |
| Decider 2B | NoulXP `causal-letters` (the run's GGUF) | 3.6 GB (bf16, the default) or 1.9 GB (q8_0) | Decider's own GGUF readout on the CPU, NoulXP's 52 requests | **52/52 cases**, max \|Δp\| 5.0e-5 |

"60 test rows" is the measurement every GGUF gets by default (gguf.FULL_VERIFY_ROWS: the first
60 rows of the test split, every decode they make). gguf.VERIFY_ROWS can lower it for bf16
alone, the merged weights exactly, as a product decision: it then caps the decodes and spreads
them over the whole test split and its answer types, and the report adds `test_rows` and
`of_test_rows` to say it is a sample. q8_0 and f16 always get 60. The measurement is the
studio's own report (`exports/gguf-<precision>.json`, and `gguf` in `noulxp-report.json`): no
NoulXP package, check or card depends on it.

60 rows did not show what q8_0 costs; 2,000 questions did. A Decider package therefore carries
the bf16 GGUF, the merged weights exactly, by default, and q8_0 only on request (`--gguf
q8_0`). At 3.6 GB a bf16 package is over the size the registry checks by itself (2.5 GB by
default, an admin setting), so an admin asks for its check, or raises that limit.

### Threads and timings of a NoulXP build

A build's thread count, N, is the container's CPU quota (cgroup `cpu.max`, floored, 8 on the A40
pods' 8.5 CPUs), else at most 8, or `SYSTEMONE_STUDIO_THREADS` (`LAYASTUDIO_THREADS`). `noulxp
check` always runs with N, and the GGUF readout too (two fewer while the float32 reference still
reads its rows beside it); so do `noulxp export` and `noulxp conformance generate` when they run one
after the other (Decider, or test rows). Without test rows, Laya and Julia record their own answers
while `noulxp export` traces the graph: the recording with N-1 threads, the export with 1, when 3
threads or more and 12 GiB of memory are free (`SYSTEMONE_STUDIO_PARALLEL_CONFORMANCE=0`, or
`LAYASTUDIO_PARALLEL_CONFORMANCE=0`, runs them one after the other). Each step's process gets its
count as `--threads` where the command takes one, and as `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `SYSTEMONE_STUDIO_THREADS` and `LAYASTUDIO_THREADS`, and `NOULXP_THREADS`
(Laya's own runtime takes no thread count; the exporter's informative graph-versus-torch comparison
reads the last three). On a CUDA machine with 24 GiB free, the GGUF's float32 reference reads its
rows on the GPU while the converter runs (`SYSTEMONE_STUDIO_SERIAL_VERIFY=1`, or
`LAYASTUDIO_SERIAL_VERIFY=1`, keeps it after). Free memory is the container's limit less what the
kernel cannot reclaim: the page cache, which the checkpoints just read and written fill, counts as
free. Each gate's decision, the gate that made it and what it read are in the report
(`steps.conformance.overlap_gate`, and the GGUF's `timings.overlap_gate`). Every step's process,
llama.cpp's converter and the GGUF's float32 reference included, ends with the job, even a job
killed outright (it exits when the job's end closes its stdin, and on Linux on PR_SET_PDEATHSIG),
and the next build or GGUF export of the run removes the scratch folders such a job left.
`noulxp-report.json` says what each step took (`steps`: seconds, CPU seconds, threads, the
container's CPU throttling, peak threads and memory of each step's process) and on what machine
(`machine`). Thread counts move answers by rounding only: Laya's and Julia's recordings at 1 thread
and at several agree within 1e-4, and the check's tolerance is 0.01.

For scale, bfloat16 alone (the same weights in PyTorch on the GPU) moves Decider's answers by up
to 0.017 against float32. A publish dry run of each fine-tune lists the maker's own files with
the package as `noulxp/` (Julia: 18 files, 1.2 GB; Decider: 35 files, 5.8 GB) and writes the
model card from the run's measurements.

### Other recipes and a yes/no task

| Run | Accuracy before → after | ECE | Log loss | McNemar (fixed / broken, p) | Training | Peak GPU memory |
|---|---|---|---|---|---|---|
| Decider 2B, Emotion, LoRA with the proper scoring rule (log + spherical score on the letters) | 82.5% → **90.7%** (88.1–92.7) | 0.021 → 0.029 | 0.499 → 0.331 | 74 / 25, p = 8.5e-7 | 114 s | 4.2 GB |
| Decider 2B, Emotion, 4-bit QLoRA (NF4 base, bitsandbytes), 1 epoch | 82.5% → 87.7% (84.8–90.1) | 0.021 → 0.040 | 0.499 → 0.427 | 62 / 31, p = 0.0017 | 152 s | 3.5 GB |
| Julia 1, prompt injection (one noul question, 116 test decisions), LoRA | 42.2% → **95.7%** (90.3–98.1) | 0.472 → 0.027 | 1.834 → 0.131 | 66 / 4, p = 1.7e-15 | 42 s | 1.4 GB |
| Decider 2B, prompt injection, LoRA (cross-entropy) | 51.7% → **98.3%** (93.9–99.5) | 0.322 → 0.017 | 0.800 → 0.238 | 55 / 1, p = 1.6e-15 | 85 s | 6.5 GB |

The proper scoring rule beat Decider's own cross-entropy on Emotion in accuracy, log loss and
calibration, so it is now the Decider trainer's default (`"objective": "ce"` keeps the original).
The 4-bit run merges its adapters into the bfloat16 base, not into the 4-bit copy, and gives a
checkpoint of the same format. On prompt injection the validation split was answered perfectly,
and the yes/no temperature fit ran to the edge of its range (0.25 then); the fit now keeps to the
Laya trainer's [0.5, 5], as laya-mlx does.

The studio's test suite ran on the pod too: 37 passed and 9 skipped (the MLX-only tests), after
one fix: with flash-linear-attention installed, transformers sends Qwen3.5's linear attention to
Triton kernels even for a model on the CPU, so the studio now puts transformers' own PyTorch
functions back on any device but CUDA. On the Mac, with MLX, 87 pass.

**Cost:** one A40 pod ($0.49 an hour) for about an hour: $0.43 of RunPod credit. It was terminated at the end.

## What comes next, in order

Effort is a working estimate for one engineer who knows this code: reading the maker's code
and licence, the trainer or prompt plug-in, a token-for-token or answer-for-answer parity test
against the maker's own code, the exports and NoulXP, and one GPU run like the one above (each
run costs well under a dollar of A40 time).

1. **The other letter-readout decoders** (letter family; the `causal-letters` profile exists).
   cua-s1-4b-0.2, Bespoke-Nimble-9B, JevK5 (2B, 4B, 9B), Jev-Style-Qwen3.5-2B, Eikos-4B. The
   Decider trainer already does LoRA, QLoRA, the letter loss, calibration, GGUF and MLX; what
   differs is each model's prompt. First split `decider.py` into a prompt plug-in (about 1 day,
   with cua-s1-4b as the second plug-in), then about half a day per model. Nimble is a LoRA
   adapter on Qwen3.5-9B (merge it first; 27 GB for LoRA, 12 GB in 4 bits). JevK5's knockout
   rounds above 16 options are inference only: train on rows of at most 16 options; its NoulXP
   package needs knockout in noulxp's causal-letters profile or a refusal above 16 (about 1
   more day, a noulxp change). Total: about 4 to 5 days.
2. **The other encoder + marker models** (laya family; `encoder-markers` exists, exporters do
   not). Von (von-sdk's model with its order-invariant option mask), open-jev and JevK5-Lite
   (DeBERTa-v3-large heads; PyTorch only, laya-mlx has no DeBERTa), dev-0.4b. About 1 day each
   for the trainer path and 0.5 to 1 day each for its `noulxp export` (a noulxp change, on its
   own branch). Total: about 5 to 6 days.
3. **Per-option cross-encoders** (crossenc). A new trainer (a sequence-classification score
   per option, softmax over a question's options, listwise loss; about 1.5 days), then
   AlexWortega/openjev, mobarmg's schema scorer and argos1111's Japanese ModernBERT (whose
   fine-tunes stay CC-BY-SA). pngwn's scorer is non-commercial: train only. The
   `encoder-pairs` profile is on noulxp's unreleased `encoder-pairs` branch, made for Mira;
   releasing it is the owner's call. About 3 to 4 days with the exporters.
4. **GLiNER2 / GLiClass** (gliner). Fastino's gliner2 package and GLiClass have their own
   trainers: wrap them (about 1 day). NoulXP has no profile for label-conditioned extractors:
   designing one (spec, reference runtime, conformance) is about 2 to 3 days in noulxp.
5. **Decoder + pointer or slot heads** (head). Kev first (its own trainer, 0.6B and 4B), then
   OpenThai (256-slot head and control tokens). Jev-Omni waits for a published trainer (12B,
   multimodal); NanoJev has no licence. A NoulXP profile for learned heads is needed. About 4
   to 5 days.
6. **Frozen embedder + heads** (CLM). Training the heads on cached embeddings is cheap (about
   1 day); serving needs the frozen Qwen3-8B; a NoulXP profile is needed (about 2 days).
7. **Tiny scorers** (cua-s1-forms, cua-s1-nano). Byte-level models under a million
   parameters, trained on a CPU; about 2 days with a small NoulXP profile.

Not planned: models whose licence forbids derivatives (Tev1, NanoJev), weights-free APIs
(TypeSafe's Jev), and SAGEA's Mira (exclusive launch, benchmark pending).
