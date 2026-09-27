"""The PyTorch trainer: the same LoRA variants as MLX, with the same numbers."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("laya")

from test_engine import QUESTIONS, checkpoint, make_rows  # noqa: E402, F401 - fixture

from layastudio import engine, torch_engine  # noqa: E402


@pytest.mark.parametrize("dora", [False, True])
@pytest.mark.parametrize("rslora", [False, True])
def test_fusion_is_exact(dora, rslora):
    torch.manual_seed(0)
    base = torch.nn.Linear(8, 6)
    lora = torch_engine.lora_linear()(base, 4, 8, 0.0, dora=dora, rslora=rslora)
    with torch.no_grad():
        lora.lora_b.normal_()
        if dora:
            lora.magnitude.mul_(torch.empty_like(lora.magnitude).uniform_(0.5, 1.5))
    lora.eval()
    x = torch.randn(3, 8)
    torch.testing.assert_close(lora(x), lora.fused(torch.float32)(x), rtol=1e-5, atol=1e-5)


def test_dora_starts_as_the_base_layer():
    torch.manual_seed(1)
    base = torch.nn.Linear(8, 6)
    lora = torch_engine.lora_linear()(base, 4, 8, 0.0, dora=True)
    x = torch.randn(3, 8)
    torch.testing.assert_close(lora(x), base(x), rtol=1e-5, atol=1e-5)


def test_dora_matches_mlx():
    """The same weights give the same outputs in both frameworks, so runs stay portable."""
    import mlx.core as mx
    import mlx.nn as mnn

    rng = np.random.default_rng(0)
    w = rng.normal(size=(6, 8)).astype(np.float32)
    b = rng.normal(size=6).astype(np.float32)
    a = rng.normal(size=(8, 4)).astype(np.float32)
    bb = rng.normal(size=(4, 6)).astype(np.float32)
    m = rng.uniform(0.5, 2.0, size=6).astype(np.float32)
    x = rng.normal(size=(3, 8)).astype(np.float32)

    t_base = torch.nn.Linear(8, 6)
    with torch.no_grad():
        t_base.weight.copy_(torch.from_numpy(w))
        t_base.bias.copy_(torch.from_numpy(b))
    t = torch_engine.lora_linear()(t_base, 4, 8, 0.0, dora=True, rslora=True)
    with torch.no_grad():
        t.lora_a.copy_(torch.from_numpy(a))
        t.lora_b.copy_(torch.from_numpy(bb))
        t.magnitude.copy_(torch.from_numpy(m))

    m_base = mnn.Linear(8, 6)
    m_base.weight, m_base.bias = mx.array(w), mx.array(b)
    ml = engine.lora_class()(m_base, 4, 8, 0.0, dora=True, rslora=True)
    ml.lora_a, ml.lora_b, ml.magnitude = mx.array(a), mx.array(bb), mx.array(m)
    ml.eval()

    np.testing.assert_allclose(
        t(torch.from_numpy(x)).detach().numpy(), np.asarray(ml(mx.array(x))), rtol=1e-4, atol=1e-4
    )


def test_variants_train_on_pytorch(checkpoint, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("LAYASTUDIO_DEVICE", "cpu")
    # transformers' ModernBERT defaults its special-token ids to the real 50k vocabulary.
    config = checkpoint / "encoder/config.json"
    ids = {
        "pad_token_id": 0,
        "cls_token_id": 2,
        "bos_token_id": 2,
        "sep_token_id": 3,
        "eos_token_id": 3,
    }
    config.write_text(json.dumps({**json.loads(config.read_text()), **ids}))
    workspace = tmp_path / "ws"
    rows = "\n".join(json.dumps(r) for r in make_rows(30))
    meta = engine.create_dataset("tiny", QUESTIONS, rows, "t.jsonl", workspace=workspace)
    hp = {
        "epochs": 1,
        "batch_size": 8,
        "precision": "float32",
        "dora": True,
        "rslora": True,
        "loraplus_ratio": 16.0,
    }
    spec = {
        "run_id": "tiny-torch",
        "dataset": meta["id"],
        "base_model": f"path:{checkpoint}",
        "baseline": False,
    }
    summary = torch_engine.fit(
        spec, {**engine.HYPERPARAMETERS, **hp}, lambda *a, **k: None, workspace
    )
    assert summary["trainable_params"] > 0
    out = workspace / "runs/tiny-torch/model"
    assert set(engine.safetensors_header(out / "model.safetensors")) == set(
        engine.safetensors_header(checkpoint / "model.safetensors")
    )
    cfg = json.loads((out / "rl_agent_config.json").read_text())
    assert cfg["fine_tuned"]["lora_variants"] == ["DoRA", "rsLoRA", "LoRA+ x16"]
