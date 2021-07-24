import os

from hydra.experimental import compose, initialize
from omegaconf import DictConfig
import pytest

config_files = [f.split(".")[0] for f in os.listdir("../conf") if "yaml" in f]


@pytest.mark.parametrize("config_name", config_files)
def test_cfg(config_name: str) -> None:
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name=config_name,
        )
        assert isinstance(cfg, DictConfig)
