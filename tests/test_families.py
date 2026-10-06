from systemone_studio.families import CATALOG, FAMILIES, assess, catalogue, find


def test_every_model_is_in_a_known_family():
    assert all(m.family in FAMILIES for m in CATALOG)


def test_nothing_is_hidden_and_big_models_warn():
    small = {f["key"]: f for f in catalogue(8.0, "cuda")["families"]}
    assert sum(len(f["models"]) for f in small.values()) == len(CATALOG)
    nimble = next(
        m for m in small["letter"]["models"] if m["repo"] == "bespokelabs/Bespoke-Nimble-9B"
    )
    assert nimble["fit"] == "too-big"
    assert any("bigger GPU" in w for w in nimble["warnings"])
    # The cloud studio trains what the studio trains: Nimble has no trainer yet, so no pointer.
    assert not any("systemonemodels.ai/studio" in w for w in nimble["warnings"])


def test_licences_warn():
    assert any(
        "Non-commercial" in w
        for w in assess(find("pngwn/system-one-qwen3.5-4b-scorer"), 64, "cuda")["warnings"]
    )
    assert any("No licence" in w for w in assess(find("together-ai/tev1"), 64, "cuda")["warnings"])
    assert assess(find("typesafe-ai/jev"), 64, "cuda")["fit"] == "not-trainable"


def test_laya_fits_a_16_gb_mac_and_finds_by_registry_name():
    laya = find("convai-innovations/laya")
    assert laya is not None and assess(laya, 11.8, "mlx")["fit"] == "fits"


def test_julia_and_decider_are_catalogued_with_their_trainers():
    from systemone_studio.families import licence_allows, trainable, trainer_status

    julia = find("supersonic-labs/julia-1")
    assert julia and julia.repo == "SupersonicLabs/Julia-1" and julia.family == "laya"
    assert julia.kind == "julia" and julia.licence == "apache-2.0"
    for size in ("decider-0.8b", "decider-2b", "decider-4b"):
        model = find(f"Mapika/{size}")
        assert model.kind == "decider" and trainer_status(model)["ready"]
    repos = {m["repo"] for m in trainable(16, "mlx")}
    assert {"SupersonicLabs/Julia-1", "Mapika/decider-2b", "convaiinnovations/laya"} <= repos
    assert not any("mira" in r.lower() for r in repos)  # SAGEA's model is not catalogued
    # Licences: derivatives allowed by open and share-alike; non-commercial trains, never published.
    assert licence_allows("open", "publish") and licence_allows("sharealike", "publish")
    assert licence_allows("noncommercial", "train") and not licence_allows(
        "noncommercial", "publish"
    )
    assert not licence_allows("none", "train") and not licence_allows("closed", "train")


def test_a_model_without_a_trainer_is_shown_with_why():
    nimble = assess(find("bespokelabs/Bespoke-Nimble-9B"), 64, "cuda")
    assert nimble["trains_here"] is False and nimble["fit"] == "fits"
    assert any("trainer for this model's own format" in w for w in nimble["warnings"])
    tev = assess(find("together-ai/tev1"), 64, "cuda")
    assert tev["trains_here"] is False
    assert any("does not allow derivatives" in w for w in tev["warnings"])


def test_decider_says_when_it_is_too_big_and_when_qlora_fits():
    decider = find("Mapika/decider-2b")
    assert assess(decider, 24, "cuda")["fit"] == "fits"
    tight = assess(decider, 8, "cuda")
    assert tight["fit"] == "qlora" and any("4-bit QLoRA" in w for w in tight["warnings"])
    small = assess(decider, 4, "cuda")
    assert small["fit"] == "too-big" and any(
        "the cloud studio at https://systemonemodels.ai/studio (coming)" in w
        for w in small["warnings"]
    )
    assert assess(find("SupersonicLabs/Julia-1"), 4, "cpu")["fit"] == "fits"
