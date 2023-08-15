"""
Generates dataset of annotated CCLIPT Embeddings
"""
import os
import pickle
from minigrid.wrappers import RGBImgObsWrapper
from nlgoals.babyai.custom.constants import NAME_TO_CLASS, SIZE_TO_ENVS

import torch
from transformers import AutoTokenizer
import numpy as np
from tqdm.auto import tqdm

from nlgoals.babyai.custom.envs import COLOR_NAMES, OBJ_MAP
from nlgoals.babyai.custom.eval_utils import run_oracle
from nlgoals.babyai.custom.constants import NAME_TO_KWARGS
from nlgoals.babyai.custom.utils import paraphrase_mission
from nlgoals.data.transforms import CLIPImageTransform
from nlgoals.models.clipt import CLIPT


def setup_clipt(args, device):
    clipt_state_dict = torch.load(args.clipt_checkpoint, map_location=device)[
        "state_dict"
    ]
    clipt = CLIPT(**args.clipt.as_dict())
    clipt.load_state_dict(clipt_state_dict, strict=False)

    clipt.to(device)
    clipt.eval()

    return clipt


def prepare_visual_input(img, img_transform, device):
    # 1 x 3 x H x W
    return img_transform(torch.from_numpy(img).unsqueeze(0)).to(device)


def prepare_textual_input(text, tokenizer, device):
    out = tokenizer(text, return_tensors="pt")
    # 1 x L
    return {
        "input_ids": out["input_ids"].to(device),
        "attention_mask": out["attention_mask"].to(device),
    }


def run_rollout(env, clipt, seed, seed_offset, img_transform, text_transform):
    goal_image, goal_text, seed = run_oracle(env, seed, seed_offset)

    # 1 x 3 x H x W
    visual_goal = prepare_visual_input(goal_image, img_transform, clipt.device)
    # 1 x L
    textual_goal = prepare_textual_input(goal_text, text_transform, clipt.device)

    obs, _ = env.reset(seed=seed)

    goal = env.unwrapped.goal_obj
    dists = env.unwrapped.distractors

    # 1 x 3 x H x W
    visual_context = prepare_visual_input(obs["image"], img_transform, clipt.device)

    # 1 X 2 X 3 X H x W
    images = torch.stack([visual_context, visual_goal], dim=1)

    visual_traj_input = {"images": images}
    visual_traj = (
        clipt.encode_visual_traj(**visual_traj_input, normalize=True)
        .squeeze()
        .cpu()
        .numpy()
    )

    textual_traj_input = {
        "images": images,
        "text_input_ids": textual_goal["input_ids"],
        "text_attn_mask": textual_goal["attention_mask"],
    }
    textual_traj = (
        clipt.encode_text_traj(**textual_traj_input, normalize=True)
        .squeeze()
        .cpu()
        .numpy()
    )

    return textual_traj, visual_traj, seed, goal, dists


def generate_env_data(
    env_name, clipt, num_rollouts, start_seed, img_transform, text_transform
):
    EnvClass = NAME_TO_CLASS[env_name]
    env_kwargs = NAME_TO_KWARGS[env_name]

    env = EnvClass(**env_kwargs, highlight=False)
    env = RGBImgObsWrapper(env, tile_size=28)

    seeds = np.linspace(
        start=start_seed,
        stop=start_seed + num_rollouts - 1,
        num=num_rollouts,
        dtype=int,
    )
    textual_traj_embs = np.empty((num_rollouts, clipt.emb_dim), dtype=np.float32)
    visual_traj_embs = np.empty((num_rollouts, clipt.emb_dim), dtype=np.float32)
    goals = []
    dists = []

    for i in tqdm(range(num_rollouts), desc="Rollouts"):
        seed = int(seeds[i])
        textual_traj_emb, visual_traj_emb, used_seed, goal, roll_dists = run_rollout(
            env=env,
            clipt=clipt,
            seed=seed,
            seed_offset=num_rollouts,
            img_transform=img_transform,
            text_transform=text_transform,
        )
        textual_traj_embs[i] = textual_traj_emb
        visual_traj_embs[i] = visual_traj_emb
        seeds[i] = used_seed
        goals.append(goal)
        dists.append(roll_dists)

    return textual_traj_embs, visual_traj_embs, seeds, goals, dists


def generate_data(clipt, num_rollouts, seed, img_transform, text_transform):
    env_names = SIZE_TO_ENVS["small"]
    name_to_id = {name: i for i, name in enumerate(env_names)}

    num_envs = len(env_names)
    textual_traj_embs = np.empty(
        (num_rollouts * num_envs, clipt.emb_dim), dtype=np.float32
    )
    visual_traj_embs = np.empty(
        (num_rollouts * num_envs, clipt.emb_dim), dtype=np.float32
    )
    task_ids = np.empty((num_rollouts * num_envs,), dtype=np.int64)
    seeds = np.empty((num_rollouts * num_envs,), dtype=np.int64)
    goals = []
    dists = []

    for i, env_name in tqdm(enumerate(env_names), desc="Envs"):
        env_output = generate_env_data(
            env_name, clipt, num_rollouts, seed, img_transform, text_transform
        )
        env_text_trajs, env_vis_trajs, env_seeds, env_goals, env_dists = env_output

        textual_traj_embs[i * num_rollouts : (i + 1) * num_rollouts] = env_text_trajs
        visual_traj_embs[i * num_rollouts : (i + 1) * num_rollouts] = env_vis_trajs
        task_ids[i * num_rollouts : (i + 1) * num_rollouts] = name_to_id[env_name]
        seeds[i * num_rollouts : (i + 1) * num_rollouts] = env_seeds
        goals.extend(env_goals)
        dists.extend(env_dists)

    return textual_traj_embs, visual_traj_embs, task_ids, seeds, goals, dists


class ParaphrasingTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text, return_tensors=None):
        para_text = paraphrase_mission(text)
        return self.tokenizer(para_text, return_tensors=return_tensors)


def main(args):
    _ = torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clipt = setup_clipt(args, device)

    # transforms
    img_transform = CLIPImageTransform(**args.img_transform.as_dict())
    tokenizer = AutoTokenizer.from_pretrained(clipt.clip_model_name)
    if args.paraphrase:
        text_transform = ParaphrasingTokenizer(tokenizer)
    else:
        text_transform = tokenizer

    # generate data
    data = generate_data(
        clipt, args.num_rollouts, args.seed, img_transform, text_transform
    )
    # np array, np array, np array, list, list
    textual_trajs, visual_trajs, task_ids, seeds, goal_objs, distractors = data

    # save data
    os.makedirs(args.save_dir, exist_ok=True)

    np.save(os.path.join(args.save_dir, "textual_trajs.npy"), textual_trajs)
    np.save(os.path.join(args.save_dir, "visual_trajs.npy"), visual_trajs)
    np.save(os.path.join(args.save_dir, "task_ids.npy"), task_ids)
    np.save(os.path.join(args.save_dir, "seeds.npy"), seeds)

    # Save lists
    with open(os.path.join(args.save_dir, "goal_objs.pkl"), "wb") as f:
        pickle.dump(goal_objs, f)
    with open(os.path.join(args.save_dir, "distractors.pkl"), "wb") as f:
        pickle.dump(distractors, f)


if __name__ == "__main__":
    import jsonargparse

    parser = jsonargparse.ArgumentParser(description=__doc__)

    parser.add_argument("--seed", type=int, default=0, help="start random seed")

    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/babyai/clipt-embs/",
        help="save directory",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=1000,
        help="Number of rollouts per environment",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum number of steps to perform per rollout",
    )

    # clipt stuff
    parser.add_class_arguments(CLIPT, "clipt")
    parser.add_argument("--clipt_checkpoint", type=str, required=True)

    parser.add_class_arguments(CLIPImageTransform, "img_transform")

    # env stuff
    parser.add_argument(
        "--env.cc.enable",
        type=bool,
        default=False,
        help="Whether to causally confuse the environment",
    )
    parser.add_argument(
        "--env.cc.obj_type",
        type=str,
        choices=list(OBJ_MAP.keys()),
        default="key",
        help="Object type to use for causal confusion",
    )
    parser.add_argument(
        "--env.cc.obj_color",
        type=str,
        choices=COLOR_NAMES,
        default="red",
        help="Object color to use for causal confusion",
    )

    args = parser.parse_args()

    main(args)
