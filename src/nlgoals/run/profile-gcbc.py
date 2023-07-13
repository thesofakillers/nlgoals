"""Profile forward and optionally backwards of GCBC model."""
import cProfile

import jsonargparse
import torch
from tqdm.auto import tqdm

from nlgoals.models.perception_encoders import VisionEncoder, ProprioEncoder
from nlgoals.models.clipt import CLIPT
from nlgoals.models.gcbc import CALVIN_GCBC


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_batches = args.num_batches
    batch_size = args.batch_size
    seq_len = args.seq_len

    model = CALVIN_GCBC(
        traj_encoder_kwargs=args.clipt.as_dict(),
        vision_encoder_kwargs=args.vision_encoder.as_dict(),
        proprio_encoder_kwargs=args.proprio_encoder.as_dict(),
        **args.gcbc.as_dict(),
    )

    model.to(device)

    batch = {
        "rgb_perc": torch.rand(batch_size, seq_len - 1, 3, 224, 224).to(device),
        "proprio_perc": torch.rand(batch_size, seq_len - 1, 8).to(device),
        "seq_lens": torch.randint(1, seq_len, (batch_size,)).to(device),
    }
    visual_goal = torch.rand(batch_size, 3, 224, 224)
    textual_goal = {
        "input_ids": torch.randint(0, 1000, (batch_size, 5)).to(device),
        "attention_mask": torch.randint(0, 2, (batch_size, 5)).to(device),
    }
    goal = visual_goal if args.mode == "visual" else textual_goal
    for _ in tqdm(range(num_batches)):
        model.forward(batch=batch, goal=goal, traj_mode=args.mode)

        if args.backward:
            raise NotImplementedError

        model.reset()


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_class_arguments(
        CALVIN_GCBC,
        "gcbc",
        skip={"vision_encoder_kwargs", "proprio_encoder_kwargs", "traj_encoder_kwargs"},
    )

    parser.add_class_arguments(CLIPT, "clipt")

    parser.add_class_arguments(VisionEncoder, "vision_encoder")
    parser.add_class_arguments(ProprioEncoder, "proprio_encoder")

    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2)
    parser.add_argument("--mode", choices=["visual", "textual"], default="visual")
    parser.add_argument("--backward", action="store_true", default=False)
    parser.add_argument("--output", type=str, default=f"profiles/gcbc.prof")

    args = parser.parse_args()

    cProfile.run("main(args)", filename=args.output, sort="cumtime")
