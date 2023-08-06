"""Profile the babyai dataset for bottlenecks"""
import cProfile

import jsonargparse

from nlgoals.data.babyai.datamodule import BabyAIDM
from nlgoals.interfaces.clipt import BABYAI_CLIPT_PREPARE_CONFIG
from nlgoals.data.transforms import TRANSFORM_MAP, TransformName


def main(args):
    # create BabyAI dataset instance

    transform_config = BABYAI_CLIPT_PREPARE_CONFIG[args.data.transform_variant]
    transform_config["mode"] = args.data.transform_variant
    if args.data.transform_name is not None:
        data_transform = TRANSFORM_MAP[args.data.transform_name.value](
            **transform_config
        )

    babyai_dm = BabyAIDM(**args.data.as_dict(), transform=data_transform)
    babyai_dm.prepare_data()
    babyai_dm.setup(stage="debug")

    dataloader = babyai_dm.train_dataloader()

    for epoch in range(3):
        print(epoch)
        for batch in dataloader:
            print(batch.keys())


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    parser = jsonargparse.ArgumentParser(description=__doc__)

    # transforms
    parser.add_argument(
        "--data.transform_name", type=TransformName, default="clipt-prepare"
    )
    parser.add_argument(
        "--data.transform_variant",
        type=str,
        default="with_clip",
        choices=["without_clip", "with_clip"],
    )

    # data
    parser.add_class_arguments(BabyAIDM, "data", skip={"transform"})

    parser.add_argument("--output", type=str, default=f"profiles/babyai.prof")

    args = parser.parse_args()

    cProfile.run("main(args)", filename=args.output, sort="cumtime")
