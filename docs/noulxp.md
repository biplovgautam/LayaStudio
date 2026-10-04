# NoulXP packages from System One Studio

[NoulXP](https://github.com/systemonemodels/noulxp) is the open standard that lets any engine run a
System One model from a package of files, with no code written for that model. A model version on
systemonemodels.tech that carries a package in a `noulxp/` folder is checked by the registry's
engine against the package's own conformance file, on the CPU; a pass shows the model as
**NoulXP compatible**.

The studio builds that package with **noulxp 0.4** (the release the registry checks with; the
`export` extra installs it) for every fine-tune of a family that has both a trainer here and a
NoulXP exporter. Today that is every fine-tune the studio writes: Laya and Julia 1
(`encoder-markers`) and Decider (`causal-letters`).

## What a fine-tune gets

```bash
uv sync --extra export
python -m layastudio.export run:<id> --target noulxp     # or Export → NoulXP package
python -m layastudio.publish_systemone run:<id>          # or Publish to System One
```

1. `noulxp export laya` (or `julia`) writes the package from the run's checkpoint: an ONNX graph
   whose weights are the checkpoint's own `model.safetensors`, the tokenizer, `template.json` and
   `calibration.json`. For a Decider fine-tune, `noulxp export decider` packages the run's GGUF
   (`layastudio/gguf.py` converts it with llama.cpp's converter at a pinned commit; q8_0 by
   default, as Decider's own package ships, `--gguf bf16` for the exact weights) with the
   tokenizer, `prompt.json` and the fine-tune's temperatures as `calibration.json`.
2. `noulxp conformance generate` records the fine-tune's own answers, from the model's own runtime
   on the CPU (the `laya` package in float32; Julia's own inference; Decider's own GGUF readout
   through llama.cpp), to NoulXP's request set (52 requests, 11 languages: the coverage SPEC.md 9.2
   asks for). Rows of the run's own test split can be added with `--test-rows N` (default 0);
   **rows added that way are published inside the package**, in `conformance.jsonl`, so a private
   dataset should stay at 0. A question over NoulXP's limits (20 options, 10 levels) cannot be asked
   through a package; it is left out of the file and the export says so.
3. `noulxp validate`, then `noulxp check` on the CPU. The package passes only when the reference
   runtime reproduces every case (each probability within 0.01, the same leading option) and the
   file covers what the standard asks.

A package that passes is kept as `runs/<id>/noulxp/`, with the check's report inside it
(`check-cpu.json`). One that fails is kept as `runs/<id>/noulxp-failed/` for reading and never
replaces the run's package. `runs/<id>/noulxp-report.json` describes the last attempt.

Publishing uploads the passing package as the version's `noulxp/` folder (the weights file is the
checkpoint's own, so the registry stores it once), and the model card gets one line saying how it
was checked. A run without a passing package gets one built and checked first; if it does not pass,
nothing is uploaded. `--skip-noulxp` (in the studio, untick *Include a NoulXP package*) publishes
the checkpoint alone, and its card says it carries no package.

## Every family, and the profile it needs

| Family | NoulXP profile | Where the exporter stands | In the studio |
|---|---|---|---|
| Encoder + option-marker head (Laya style) | `encoder-markers` (ONNX) | `noulxp export laya` and `noulxp export julia`, released in noulxp 0.4 | built, checked and published with every Laya and Julia 1 fine-tune; the check decides. Von, open-jev, JevK5-Lite and dev-0.4b need exporters of their own |
| Decoder, letter readout (Jev / SemIf style: Nimble, Decider, JevK5, Tev1, cua-s1, Eikos, …) | `causal-letters` (GGUF) | noulxp 0.4 exports Decider (`noulxp export decider`) and AnyJev (`noulxp export anyjev`) | built, checked and published with every Decider fine-tune; the other models' prompts arrive with their trainers |
| Per-option cross-encoder | `encoder-pairs` | exists only on an unreleased noulxp branch (`encoder-pairs`, NoulXP 0.3 draft: Mira) | arrives with this family's trainer and a noulxp release that has the profile |
| Decoder + learned pointer or slot head (Kev, OpenThai, Jev-Omni, NanoJev) | none yet | — | needs a profile first |
| GLiNER2 / GLiClass extractor | none yet | — | needs a profile first |
| Frozen embedder + small heads (CLM) | none yet | — | needs a profile first |
| Tiny scorer from scratch (cua-s1-forms, cua-s1-nano) | none yet | — | needs a profile first |

Every kind the studio trains (Laya, Julia 1, Decider) has an exporter, so every run gets a
package; a run is packaged as what its checkpoint is, whatever its base's catalogue entry says. The
Models page shows each family's NoulXP state, and [families.md](families.md) has the packages
measured on a GPU (52/52 for Julia 1 and Decider fine-tunes).
