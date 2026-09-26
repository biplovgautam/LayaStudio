from layastudio.families import CATALOG, FAMILIES, assess, catalogue, find


def test_every_model_is_in_a_known_family():
    assert all(m.family in FAMILIES for m in CATALOG)


def test_nothing_is_hidden_and_big_models_warn():
    small = {f["key"]: f for f in catalogue(8.0, "cuda")["families"]}
    assert sum(len(f["models"]) for f in small.values()) == len(CATALOG)
    nimble = next(m for m in small["letter"]["models"] if m["repo"] == "bespokelabs/Bespoke-Nimble-9B")
    assert nimble["fit"] == "too-big"
    assert any("studio.systemonemodels.tech" in w for w in nimble["warnings"])


def test_licences_warn():
    assert any("Non-commercial" in w for w in assess(find("pngwn/system-one-qwen3.5-4b-scorer"), 64, "cuda")["warnings"])
    assert any("No licence" in w for w in assess(find("together-ai/tev1"), 64, "cuda")["warnings"])
    assert assess(find("typesafe-ai/jev"), 64, "cuda")["fit"] == "not-trainable"


def test_laya_fits_a_16_gb_mac_and_finds_by_registry_name():
    laya = find("convai-innovations/laya")
    assert laya is not None and assess(laya, 11.8, "mlx")["fit"] == "fits"
