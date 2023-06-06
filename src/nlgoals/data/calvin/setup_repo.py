"""Script for setting up repo dataset"""
import hydra
import os
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./repo/conf/", config_name="datamodule")
def main(cfg: DictConfig):
    datamodule = hydra.utils.instantiate(
        cfg, training_repo_root="/Users/thesofakillers/repos/thesis/"
    )

    datamodule.prepare_data()
    datamodule.setup()


if __name__ == "__main__":
    main()
