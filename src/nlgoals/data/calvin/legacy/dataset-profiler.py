"""
Profiles CALVIN dataset for bottlenecks.
To be run with python -m cProfile -s cumtime {script-name}.py --args
"""
import jsonargparse
from nlgoals.data.calvin.legacy.datamodule import CALVINDM
from nlgoals.data.transforms import TRANSFORM_MAP, TransformName
# note: this no longer works
from nlgoals.data.calvin.transform_configs import CLIPTPrepareForCALVIN


def main(args):
    # create a CALVIN dataset instance
    if args.data.transform_name is not None:
        data_transform = TRANSFORM_MAP[args.data.transform_name.value](
            **args.data.transform_kwargs
        )
    else:
        data_transform = None
    calvin_dm = CALVINDM(**args.data.as_dict(), transform=data_transform)
    calvin_dm.prepare_data()
    calvin_dm.setup(stage="debug")

    # get a DataLoader instance to iterate over the dataset
    dataloader = calvin_dm.train_debug_dataloader()

    for epoch in range(3):
        print(epoch)
        for batch in dataloader:
            print(batch.keys())


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_class_arguments(CALVINDM, "data", skip={"transform"})
    parser.add_argument(
        "--data.transform_name", type=TransformName, default="clipt-prepare"
    )
    parser.add_dataclass_arguments(CLIPTPrepareForCALVIN, "data.transform_kwargs")

    args = parser.parse_args()

    main(args)
