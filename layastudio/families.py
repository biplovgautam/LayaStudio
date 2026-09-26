"""Every System One model family the studio knows, and whether this machine can tune it.

A System One model reads a state and answers typed questions (choice, score, noul) in
one pass. Under that shared contract the published models are built in seven different
ways, and each way needs its own trainer. This module is the catalogue: the families,
the models in them, their licences, and a memory estimate for fine-tuning, checked
against the machine the studio is running on.

Nothing is hidden. A model that will not fit gets a warning that says so and what to
do — a bigger GPU, 4-bit QLoRA, or the cloud studio — and a model whose licence limits
what may be done with a fine-tune says that too.

Sources: each model's Hugging Face card and API record, surveyed 2026-09-26.
"""

from dataclasses import asdict, dataclass, field

CLOUD_STUDIO = "https://studio.systemonemodels.tech"

# trainer: ready (train it here today) | next (being built) | planned (import works; training later)
FAMILIES = {
    "laya": {
        "name": "Encoder + option-marker head (Laya style)",
        "how": "One [MASK] per option in a bidirectional encoder; a small head scores the markers. "
        "LoRA on the encoder, the head trained in full.",
        "trainer": "ready",
        "backends": "MLX on Apple silicon; PyTorch on NVIDIA, AMD, Intel or CPU",
    },
    "letter": {
        "name": "Decoder, letter readout (Jev / SemIf style)",
        "how": "Options are lettered in the prompt; the answer is the softmax over the next-token "
        "logits of those letters. LoRA (or 4-bit QLoRA) with cross-entropy over the letters.",
        "trainer": "next",
        "backends": "PyTorch + PEFT on NVIDIA, AMD, Intel or CPU; MLX-LM on Apple silicon",
    },
    "crossenc": {
        "name": "Per-option cross-encoder",
        "how": "Each (state, question, option) gets one score from a sequence-classification head; "
        "softmax over a question's options. LoRA with a listwise loss.",
        "trainer": "next",
        "backends": "PyTorch + PEFT",
    },
    "head": {
        "name": "Decoder + learned pointer or slot head",
        "how": "A decoder backbone with its own readout head and special tokens (Kev, OpenThai, "
        "Jev-Omni, NanoJev). Each is trained with its maker's own trainer.",
        "trainer": "planned",
        "backends": "PyTorch (each model's own trainer)",
    },
    "gliner": {
        "name": "GLiNER2 / GLiClass label-conditioned extractor",
        "how": "Labels given at call time are matched against spans of the text.",
        "trainer": "planned",
        "backends": "gliner2 / gliclass packages (PyTorch)",
    },
    "embedder": {
        "name": "Frozen embedder + small heads",
        "how": "A frozen decoder embeds the state and each candidate; small heads are trained on "
        "the embeddings (CLM).",
        "trainer": "planned",
        "backends": "PyTorch",
    },
    "tiny": {
        "name": "Tiny scorer trained from scratch",
        "how": "A byte-level encoder of well under a million parameters (cua-s1). Trains on a CPU.",
        "trainer": "planned",
        "backends": "PyTorch, CPU",
    },
}

# licence kinds: open | sharealike | noncommercial | none (no licence: all rights reserved) | closed (no weights)
LICENCE_NOTE = {
    "sharealike": "Share-alike licence: fine-tunes must be published under the same licence.",
    "noncommercial": "Non-commercial licence: fine-tune for research or personal use, not for a product.",
    "none": "No licence is published, so all rights are reserved: you may experiment privately, "
    "but do not publish or ship a fine-tune.",
    "closed": "No weights are published: this model can only be called through its API.",
}


@dataclass
class CatalogModel:
    repo: str
    maker: str
    family: str
    params_b: float  # billions
    licence: str
    licence_kind: str
    registry: str | None = None  # namespace/name on systemonemodels.tech
    note: str = ""
    aliases: list[str] = field(default_factory=list)


CATALOG = [
    # Encoder + option-marker head
    CatalogModel(
        "convaiinnovations/laya",
        "Convai Innovations",
        "laya",
        0.421,
        "apache-2.0",
        "open",
        "convai-innovations/laya",
        "English base; the multilingual (322M) and typed-decisions checkpoints are subfolders.",
    ),
    CatalogModel(
        "aac6fef/laya-mlx",
        "aac6fef",
        "laya",
        0.421,
        "apache-2.0",
        "open",
        "aac6fef/laya-mlx",
        "MLX port of Laya.",
    ),
    CatalogModel(
        "aac6fef/laya-multilingual-mlx",
        "aac6fef",
        "laya",
        0.322,
        "apache-2.0",
        "open",
        "aac6fef/laya-multilingual-mlx",
        "MLX port of the multilingual checkpoint.",
    ),
    CatalogModel(
        "aac6fef/laya-typed-decisions-mlx",
        "aac6fef",
        "laya",
        0.421,
        "apache-2.0",
        "open",
        None,
        "MLX port of the typed-decisions checkpoint.",
    ),
    CatalogModel(
        "wfzyx/von",
        "wfzyx",
        "laya",
        0.395,
        "apache-2.0",
        "open",
        "wfzyx/von",
        "ModernBERT-large with an order-invariant option mask; needs von's own trainer.",
    ),
    CatalogModel(
        "com-kotobalabs/open-jev-deberta-v3-large",
        "Kotoba Labs",
        "laya",
        0.435,
        "apache-2.0",
        "open",
    ),
    CatalogModel("mpnikhil/dev-0.4b", "mpnikhil", "laya", 0.395, "apache-2.0", "open"),
    CatalogModel(
        "alibiserikbay/JevK5-Lite",
        "alibi-serikbay",
        "laya",
        0.434,
        "apache-2.0",
        "open",
        None,
        "DeBERTa-v3-large with a label-marker head.",
    ),
    # Decoder, letter readout
    CatalogModel(
        "bespokelabs/Bespoke-Nimble-9B",
        "Bespoke Labs",
        "letter",
        9.65,
        "apache-2.0",
        "open",
        "bespoke-labs/bespoke-nimble-9b",
        "LoRA adapter on Qwen3.5-9B; option codes A–Z, AA… up to 255.",
    ),
    CatalogModel(
        "Mapika/decider-2b",
        "mapika",
        "letter",
        1.88,
        "apache-2.0",
        "open",
        "mapika/decider",
        "Several answer slots per prompt.",
        ["mapika/decider-2b"],
    ),
    CatalogModel("Mapika/decider-0.8b", "mapika", "letter", 0.8, "apache-2.0", "open"),
    CatalogModel("Mapika/decider-4b", "mapika", "letter", 4.0, "apache-2.0", "open"),
    CatalogModel(
        "alibiserikbay/JevK5",
        "alibi-serikbay",
        "letter",
        4.21,
        "apache-2.0",
        "open",
        "alibi-serikbay/jevk5",
        "Knockout rounds above 16 options.",
        ["alibiserikbay/jevk5"],
    ),
    CatalogModel("alibiserikbay/JevK5-2B", "alibi-serikbay", "letter", 2.0, "apache-2.0", "open"),
    CatalogModel("alibiserikbay/JevK5-9B", "alibi-serikbay", "letter", 9.0, "apache-2.0", "open"),
    CatalogModel(
        "cua-ai/cua-s1-4b-0.2",
        "Cua",
        "letter",
        4.0,
        "apache-2.0",
        "open",
        None,
        "LoRA on Qwen3.5-4B.",
    ),
    CatalogModel(
        "togethercomputer/Tev1-4B-experimental",
        "Together AI",
        "letter",
        4.66,
        "none",
        "none",
        "together-ai/tev1",
        "",
        ["togethercomputer/tev1-4b-experimental"],
    ),
    CatalogModel(
        "togethercomputer/Tev1-0.8B-experimental", "Together AI", "letter", 0.87, "none", "none"
    ),
    CatalogModel(
        "chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2",
        "chaoliangUNSW",
        "letter",
        2.0,
        "apache-2.0",
        "open",
    ),
    CatalogModel("caiovicentino1/Eikos-4B", "caiovicentino1", "letter", 4.0, "mit", "open"),
    CatalogModel("openjev/openjev", "openjev", "letter", 27.0, "cc-by-nc-4.0", "noncommercial"),
    # Per-option cross-encoder
    CatalogModel(
        "pngwn/system-one-qwen3.5-4b-scorer",
        "pngwn",
        "crossenc",
        4.24,
        "cc-by-nc-4.0",
        "noncommercial",
        "pngwn/system-one-qwen3-5-4b-scorer",
    ),
    CatalogModel(
        "AlexWortega/openjev",
        "AlexWortega",
        "crossenc",
        2.0,
        "mit",
        "open",
        None,
        "NLI sequence classifiers in several Qwen3.5 sizes.",
    ),
    CatalogModel(
        "mobarmg/jev-schema-scorer-deberta-v3-large", "mobarmg", "crossenc", 0.435, "mit", "open"
    ),
    CatalogModel(
        "argos1111/modernbert-ja-310m-jev",
        "argos1111",
        "crossenc",
        0.31,
        "cc-by-sa-4.0",
        "sharealike",
    ),
    # Decoder + learned head
    CatalogModel(
        "jaredpalmer/kev-4b",
        "Jared Palmer",
        "head",
        4.0,
        "apache-2.0",
        "open",
        "jared-palmer/kev",
        "Pointer head over option spans; Kev's own trainer. Also 0.5B–27B sizes.",
    ),
    CatalogModel("jaredpalmer/kev-0.6b", "Jared Palmer", "head", 0.6, "apache-2.0", "open"),
    CatalogModel(
        "iapp/OpenThai-SystemOne",
        "iApp Technology",
        "head",
        0.75,
        "apache-2.0",
        "open",
        "iapp-technology/openthai-systemone",
        "256-slot head and control tokens; Thai continued pretraining.",
        ["iapp/openthai-systemone"],
    ),
    CatalogModel(
        "akhilaaa3/Jev-Omni",
        "akhilaaa3",
        "head",
        11.96,
        "apache-2.0",
        "open",
        "akhilaaa3/jev-omni",
        "Multimodal Gemma 4 12B with a 256-slot head; no published trainer.",
        ["akhilaaa3/jev-omni"],
    ),
    CatalogModel(
        "C-Tianyu/NanoJev",
        "tianyu-codings",
        "head",
        0.6,
        "none",
        "none",
        "tianyu-codings/nanojev",
        "",
        ["c-tianyu/nanojev"],
    ),
    # GLiNER2
    CatalogModel(
        "fastino/GLiNER2.5-Decide",
        "Fastino Labs",
        "gliner",
        0.486,
        "apache-2.0",
        "open",
        "fastino-labs/gliner2-5-decide",
        "",
        ["fastino/gliner2.5-decide"],
    ),
    CatalogModel(
        "fastino/GLiNER2.5-Decide-1B", "Fastino Labs", "gliner", 1.19, "apache-2.0", "open"
    ),
    CatalogModel(
        "heman10x/rlcd-modernbert-151m",
        "heman10x",
        "gliner",
        0.151,
        "apache-2.0",
        "open",
        None,
        "GLiClass, 25 slots.",
    ),
    # Frozen embedder + heads
    CatalogModel(
        "Contrastive-LM/CLM-v0.1-8B",
        "Contrastive LM",
        "embedder",
        8.0,
        "apache-2.0",
        "open",
        "contrastive-lm/clm",
        "Heads only; embeddings come from a frozen Qwen3-8B.",
        ["contrastive-lm/clm-v0.1-8b"],
    ),
    # Tiny from scratch
    CatalogModel(
        "cua-ai/cua-s1-forms",
        "Cua",
        "tiny",
        0.000706,
        "mit",
        "open",
        "cua/cua-s1-forms",
        "706K parameters.",
    ),
    CatalogModel("cua-ai/cua-s1-nano-0.1", "Cua", "tiny", 0.000855, "apache-2.0", "open"),
    # No weights
    CatalogModel(
        "typesafe/jev",
        "TypeSafe AI",
        "letter",
        0.0,
        "proprietary",
        "closed",
        "typesafe-ai/jev",
        "Hosted API only.",
    ),
]


def memory_needed_gb(model: CatalogModel) -> dict:
    """Rough memory to fine-tune, in GB: LoRA in bf16, and 4-bit QLoRA where it applies.

    Rules of thumb from the published runs (Laya on 2×T4, Kev-4B at 24.6 GB on an H100,
    Nimble on one H100): bf16 weights plus adapters, optimizer state and activations at
    micro-batch 1–2 with gradient checkpointing and up to 2k tokens.
    """
    p = model.params_b
    if model.family in ("laya", "gliner", "crossenc") and p < 1.5:
        return {"lora": round(max(3.0, p * 9 + 1.5), 1), "qlora": None}
    if model.family == "tiny":
        return {"lora": 1.0, "qlora": None}
    if model.family == "embedder":
        return {"lora": round(p * 2.2 + 2, 1), "qlora": round(p * 0.75 + 2, 1)}
    return {"lora": round(p * 2.2 + 6, 1), "qlora": round(p * 0.75 + 5, 1)}


def assess(model: CatalogModel, training_memory_gb: float | None, accelerator: str | None) -> dict:
    """What this machine can do with this model: fits, tight, too-big, or not-trainable, and why."""
    family = FAMILIES[model.family]
    needed = memory_needed_gb(model)
    warnings = []
    if model.licence_kind in LICENCE_NOTE:
        warnings.append(LICENCE_NOTE[model.licence_kind])
    if model.licence_kind == "closed":
        return {"fit": "not-trainable", "needed_gb": needed, "warnings": warnings}
    have = training_memory_gb or 0.0
    best = needed["lora"] if have >= needed["lora"] or needed["qlora"] is None else needed["qlora"]
    if not have:
        fit = "unknown"
    elif have >= needed["lora"]:
        fit = "fits"
    elif needed["qlora"] is not None and have >= needed["qlora"]:
        fit = "qlora"
        warnings.append(
            f"Needs about {needed['lora']:g} GB for LoRA; this machine has {have:g} GB, so it trains in "
            f"4-bit QLoRA (about {needed['qlora']:g} GB), which is slower and slightly less accurate."
        )
    else:
        fit = "too-big"
        warnings.append(
            f"Needs about {best:g} GB to fine-tune; this machine has {have:g} GB. Use a machine with a "
            f"bigger GPU, or the cloud studio at {CLOUD_STUDIO} (coming)."
        )
    if accelerator == "cpu" and model.params_b >= 1.0 and fit != "too-big":
        warnings.append("No GPU was found: a model this size trains very slowly on the CPU.")
    if family["trainer"] != "ready":
        warnings.append(
            "Import works now; training for this family is "
            + ("being built next." if family["trainer"] == "next" else "planned.")
        )
    return {"fit": fit, "needed_gb": needed, "warnings": warnings}


def catalogue(training_memory_gb: float | None = None, accelerator: str | None = None) -> dict:
    """The families and their models, each assessed for this machine."""
    return {
        "cloud_studio": CLOUD_STUDIO,
        "families": [
            {
                "key": key,
                **meta,
                "models": [
                    {**asdict(m), **assess(m, training_memory_gb, accelerator)}
                    for m in CATALOG
                    if m.family == key
                ],
            }
            for key, meta in FAMILIES.items()
        ],
    }


def find(repo: str) -> CatalogModel | None:
    wanted = repo.lower()
    for model in CATALOG:
        if (
            model.repo.lower() == wanted
            or wanted in model.aliases
            or (model.registry or "").lower() == wanted
        ):
            return model
    return None


# ----------------------------------------------------------------------------- imports

LAYA_ARCHITECTURES = ("laya", "modernbert", "mmbert", "von")


def family_of(item: dict) -> str | None:
    """The family of a model listed on systemonemodels.tech, from the catalogue or its manifest."""
    for key in (item.get("full_name"), item.get("hub_repo")):
        if key and (known := find(key)):
            return known.family
    architecture = (item.get("architecture") or "").lower()
    if architecture in LAYA_ARCHITECTURES:
        return "laya"
    if "gliner" in architecture:
        return "gliner"
    if architecture in ("jev", "qwen", "qwen3", "qwen3.5", "decider", "tev1", "jevk5", "nimble"):
        return "letter"
    return None


def registry_search(
    query: str | None, training_memory_gb: float | None, accelerator: str | None, limit: int = 30
):
    """Models on systemonemodels.tech, anonymously, each with its family and fit."""
    try:
        from systemone.client import Client
        from systemone.config import load
    except ImportError as error:
        raise RuntimeError("The systemone package is not installed") from error
    settings = load()
    settings.token = None  # public listing only: nothing private leaks into the page
    with Client(settings) as registry:
        found = registry.search(query or None, limit=limit, sort="downloads")
    items = []
    for item in found.get("items", []):
        family = family_of(item)
        params = (item.get("parameters") or 0) / 1e9
        known = find(item["full_name"]) or (
            find(item["hub_repo"]) if item.get("hub_repo") else None
        )
        model = CatalogModel(
            item["full_name"],
            item.get("owner_display_name") or item["namespace"],
            family or "laya",
            params or (known.params_b if known else 0.421),
            item.get("license") or (known.licence if known else "unknown"),
            # The catalogue knows what a licence permits; the registry only has its name.
            known.licence_kind
            if known
            else ("none" if item.get("license") in (None, "other") else "open"),
        )
        assessed = (
            assess(model, training_memory_gb, accelerator)
            if family
            else {"fit": "unknown", "warnings": []}
        )
        if not family:
            assessed["warnings"] = ["The studio does not recognise this model's family yet."]
        items.append(
            {
                "repo": item["full_name"],
                "summary": item.get("summary"),
                "maker": item.get("owner_display_name") or item["namespace"],
                "family": family,
                "availability": item.get("availability"),
                "parameters": item.get("parameters"),
                "license": item.get("license"),
                "downloads": item.get("downloads"),
                **assessed,
            }
        )
    return items


def imports(workspace):
    from .engine import read_json

    return read_json(workspace / "imports.json", []) or []


def import_from_registry(repo, emit, workspace, version=None):
    """Download a model from systemonemodels.tech with the CLI's own client and register it
    as a base the studio can train from."""
    from systemone.transfer import snapshot_download

    from .engine import now, write_json

    emit("phase", phase="download", message=f"Downloading {repo} from systemonemodels.tech")
    root = snapshot_download(repo, version=version)
    laya_ready = all(
        (root / name).is_file()
        for name in (
            "model.safetensors",
            "rl_agent_config.json",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
        )
    )
    entry = {
        "ref": f"path:{root}",
        "repo": repo,
        "source": "systemonemodels.tech",
        "family": "laya" if laya_ready else None,
        "trainable": laya_ready,
        "path": str(root),
        "created": now(),
    }
    known = [e for e in imports(workspace) if e["repo"] != repo]
    write_json(workspace / "imports.json", [*known, entry])
    emit("log", message=f"Imported {repo} into {root}")
    return entry
