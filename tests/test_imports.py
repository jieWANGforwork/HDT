def test_import_core_modules():
    import hdt
    from hdt.config import load_config
    from hdt.datasets.rewards import compute_diversity_returns, compute_novelty_returns
    from hdt.models import HDT, TransformerVariant

    assert hdt.__version__
    assert load_config
    assert HDT
    assert TransformerVariant
    assert compute_novelty_returns
    assert compute_diversity_returns


def test_config_loads_default():
    from hdt.config import load_config

    config = load_config("configs/default.yaml")
    assert config["model"]["mode"] == "multi_objective"
    assert config["model"]["item_num"] == 100

