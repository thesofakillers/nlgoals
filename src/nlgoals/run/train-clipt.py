"""Training of CLIPh: Contrastive Language Image Pretraining for Trajectories"""
import jsonargparse

import calvin_agent.datasets.calvin_data_module as CALVIN

from nlgoals.models.clipt import CLIPT
from nlgoals.trainer import TrainerConfig


def train(args):
    """
    Sets up dataloader
    Instantiates model
    Trains model using contrastive loss between image traj and text pairs
    """
    calvin_dm = CALVIN(**args.data.as_dict())
    model = CLIPT(**args.clipts.as_dict())

    for epoch in range(args.trainer.n_epochs):
        print(f"Epoch {epoch + 1} of {args.trainer.n_epochs}")
        # TODO
        pass


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_class_arguments(CLIPT, "clipt")
    parser.add_argument("--trainer", type=TrainerConfig, default=TrainerConfig())
    args = parser.parse_args()
    print(args)
    # train(args)
