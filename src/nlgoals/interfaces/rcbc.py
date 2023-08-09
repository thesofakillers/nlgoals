import torch

def babyai_obs_prepare(obs, transform, device):
    """
    Prepares an observation from the BabyAI environment .step() so that
    it can be passed to GCBC.step()

    Args
        obs: Dict with following keys. It's 1 x 1 because batch size 1, single timestep
            - 'image': np.ndarray (H x W x 3) between 0 and 255
            - 'direction' : np.int64, between 0 and 3 -> right, down, left, up
        transform: Transform to apply to the image
        device: the device to put the resulting tensors on
    Returns
        Dict, with the following keys
            - "rgb_perc": 1 x 1 x 3 x H x W, RGB frames of perceived state
            - "proprio_perc": 1 x 1 x 1, proprioceptive state
            - "seq_lens": 1, sequence lengths (will just be 1)
    """
    images = torch.from_numpy(obs["image"]).unsqueeze(0).to(device)
    return {
        "rgb_perc": transform(images).unsqueeze(0).to(device),
        "proprio_perc": torch.tensor([obs["direction"]])
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device),
        "seq_lens": torch.tensor([1]).to(device),
    }
