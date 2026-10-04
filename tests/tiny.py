"""Tiny random checkpoints of the new kinds, in their makers' own file layouts.

Nothing is downloaded and nothing real runs: a Julia 1 checkpoint is a 2-layer ModernBERT of
width 64 with Laya's decision head, saved as Supersonic Labs save theirs; a Decider checkpoint
is a 2-layer Qwen3.5 (one Gated DeltaNet layer, one attention layer) with a byte-level BPE
tokenizer whose 255 option labels (A..Z, AA..) are single tokens, as Qwen's are.
"""

import json
import string
from pathlib import Path

WORDS = ["alpha", "beta", "gamma", "low", "mid", "high", "red", "green", "blue", "yes", "no"]


# ----------------------------------------------------------------------------- Julia 1


def julia_checkpoint(path, seed=3):
    """A Julia 1 folder: julia_config.json, inference-policy.json, config.json, encoder/,
    tokenizer/ and float32 model.safetensors with Laya's parameter names."""
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import ModernBertConfig, ModernBertModel

    from layastudio import julia

    path = Path(path)
    (path / "encoder").mkdir(parents=True)
    (path / "tokenizer").mkdir()
    specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"]
    vocab = {
        t: i
        for i, t in enumerate(
            specials + WORDS + [":", "?", "question", "choice", "score", "noul", "false", "true"]
        )
    }
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer/tokenizer.json"))
    (path / "tokenizer/tokenizer_config.json").write_text(
        json.dumps(
            {
                "cls_token": "<bos>",
                "sep_token": "<eos>",
                "mask_token": "<mask>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        )
    )
    encoder = {
        "model_type": "modernbert",
        "architectures": ["ModernBertForMaskedLM"],
        "vocab_size": 64,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "local_attention": 16,
        "global_attn_every_n_layers": 3,
        "max_position_embeddings": 256,
        "hidden_activation": "gelu",
        "pad_token_id": 0,
        "eos_token_id": 1,
        "bos_token_id": 2,
        "cls_token_id": 2,
        "sep_token_id": 1,
        "layer_types": ["full_attention", "sliding_attention"],
        "rope_parameters": {
            "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 160000.0, "rope_type": "default"},
        },
    }
    (path / "encoder/config.json").write_text(json.dumps(encoder))
    (path / "julia_config.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "architecture": "JuliaDecisionModel",
                "weight_dtype": "float32",
                "head_layers": 2,
                "n_act": 2,
                "dropout": 0.1,
            }
        )
    )
    (path / "inference-policy.json").write_text(
        json.dumps({"max_length": 128, "head_length": 48, "strict_encoding": True, "step": 1})
    )
    (path / "config.json").write_text(json.dumps({"architecture": "JuliaDecisionModel"}))
    torch.manual_seed(seed)
    config = ModernBertConfig(**{k: v for k, v in encoder.items() if k != "architectures"})
    model = _julia_network(torch, ModernBertModel(config))
    save_file(
        {k: v.detach().float().contiguous() for k, v in model.state_dict().items()},
        str(path / "model.safetensors"),
        metadata={"format": "pt", "family": "julia"},
    )
    assert julia.config(path)["max_len"] == 128
    return path


def _julia_network(torch, encoder):
    from laya.common import DecisionModel

    model = DecisionModel(encoder, 2, 2, 0.1)
    with torch.no_grad():
        # transformers initialises ModernBERT at std 0.02, which leaves every marker of a random
        # model with the same state; weights at the scale of a trained model mix the options in.
        for parameter in model.encoder.parameters():
            if parameter.dim() > 1:
                parameter.normal_(0, parameter.shape[1] ** -0.5)
        model.scorer[3].weight.normal_(0, 0.5)
    return model


# ----------------------------------------------------------------------------- Decider


def label_tokenizer():
    """A byte-level BPE tokenizer in which every one- and two-letter uppercase label is one
    token, the way Qwen's tokenizer has them, and a few words are whole tokens."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel

    alphabet = ByteLevel.alphabet()
    vocab = {c: i for i, c in enumerate(sorted(alphabet))}
    merges = []

    def add(left, right):
        merged = left + right
        if merged not in vocab:
            vocab[merged] = len(vocab)
            merges.append((left, right))

    upper = string.ascii_uppercase
    for a in upper:
        for b in upper:
            add(a, b)
    for word in ["Context", "Question", "Options", "Answer", "Proposed", *WORDS]:
        pieces = list(word)
        while len(pieces) > 1:
            add(pieces[0], pieces[1])
            pieces = [pieces[0] + pieces[1], *pieces[2:]]
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["<|endoftext|>"])
    return tokenizer


def decider_checkpoint(path, seed=0, isolated=True, tokenizer_dir=None):
    """A Decider folder: config.json (Qwen3_5ForCausalLM), model.safetensors, tokenizer.json,
    tokenizer_config.json and decider_config.json (plain layout, isolated levels).

    tokenizer_dir: a real Decider tokenizer (tokenizer.json and tokenizer_config.json) to use
    instead of the tiny one, for llama.cpp's converter, which knows Qwen's tokenizer only."""
    import shutil

    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    path = Path(path)
    path.mkdir(parents=True)
    if tokenizer_dir is not None:
        from tokenizers import Tokenizer

        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copy(Path(tokenizer_dir) / name, path / name)
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        vocab = 248320
    else:
        tokenizer = label_tokenizer()
        tokenizer.save(str(path / "tokenizer.json"))
        (path / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "eos_token": "<|endoftext|>",
                    "pad_token": "<|endoftext|>",
                    "bos_token": None,
                    "add_prefix_space": False,
                }
            )
        )
        vocab = 1024
    eos = tokenizer.token_to_id("<|endoftext|>")
    config = Qwen3_5TextConfig(
        vocab_size=vocab,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        full_attention_interval=2,
        tie_word_embeddings=True,
        max_position_embeddings=4096,
        eos_token_id=eos,
        pad_token_id=None,
    )
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(config).float().eval()
    with torch.no_grad():  # letters with distinct, non-trivial logits
        model.model.embed_tokens.weight.normal_(0, 0.5)
    model.save_pretrained(str(path), safe_serialization=True)
    (path / "decider_config.json").write_text(
        json.dumps(
            {
                "temperature": 1.2,
                "temperature_by_type": {"choice": 1.1, "noul": 1.5, "score": 1.3},
                "version": "tiny-v1",
                "layout": "plain",
                "max_options": 255,
                "max_state_tokens": 32768,
                "isolated_levels": isolated,
            }
        )
    )
    return path
