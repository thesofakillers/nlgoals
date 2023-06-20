import jsonargparse
import cProfile

import hydra
from tqdm import tqdm
from pytorch_lightning.trainer.supporters import CombinedLoader

from nlgoals.interfaces.gcbc import (
    calvin_gcbc_collate,
    calvin_gcbc_textual,
    calvin_gcbc_visual,
)


def main(args):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module(config_module="nlgoals.data.calvin.repo.conf")
    datamodule_cfg = hydra.compose(
        config_name=args.data.config_name,
        overrides=[] if args.data.shared_memory is True else ["datasets=vision_lang"],
    )
    datamodule_cfg.batch_size = args.data.batch_size
    datamodule_cfg.num_workers = args.data.num_workers
    datamodule_cfg.root_data_dir = args.data.data_dir
    datamodule = hydra.utils.instantiate(datamodule_cfg)
    datamodule.collator.custom_collate_fn = calvin_gcbc_collate

    datamodule.prepare_data()
    datamodule.setup()

    train_loaders = datamodule.train_dataloader()
    combined_train_loader = CombinedLoader(train_loaders, "max_size_cycle")

    num_batches = len(combined_train_loader)

    num_iterations = 10
    for i, batch in tqdm(enumerate(combined_train_loader), total=num_iterations):
        type(batch)
        if i == num_iterations:
            break


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data.config_name", type=str, required=True, default="datamodule.yaml"
    )
    parser.add_argument("--data.batch_size", type=int, default=32)
    parser.add_argument("--data.num_workers", type=int, default=18)
    parser.add_argument(
        "--data.data_dir", type=str, required=True, help="Must be absolute path"
    )
    parser.add_argument("--data.shared_memory", type=bool, default=True)
    parser.add_argument("-o", "--output", type=str)

    args = parser.parse_args()

    # main(args)
    cProfile.run("main(args)", filename=args.output)
