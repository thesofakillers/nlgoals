import pytest

from torch.utils.data import DataLoader
import torch

from nlgoals.data.calvin import CALVIN


@pytest.fixture
def frame_keys():
    frame_keys = ["rgb_static", "actions", "rel_actions"]
    return frame_keys


@pytest.fixture
def calvin(frame_keys):
    data_dir = "data/calvin/calvin_debug_dataset/"
    split = "training"
    num_frames = 5
    dataset = CALVIN(data_dir, split, num_frames, frame_keys)
    return dataset


def test_length(calvin):
    # we know that the debug dset has 5124 samples
    assert len(calvin) == 9


def test_getitem(calvin):
    item = calvin[0]
    assert isinstance(item, dict)

    expected_keys = ["lang_ann", "task"] + calvin.frame_keys
    assert all(k in item for k in expected_keys)


def test_dataloading(calvin):
    batch_size = 4
    dataloader = DataLoader(calvin, batch_size=batch_size, drop_last=True)

    for batch in dataloader:
        assert isinstance(batch, dict)

        expected_keys = ["lang_ann", "task"] + calvin.frame_keys
        assert all(k in batch for k in expected_keys)

        for frame_key in calvin.frame_keys:
            assert frame_key in batch
            assert isinstance(batch[frame_key], torch.Tensor)
            assert batch[frame_key].shape[1] == calvin.num_frames
            assert (
                batch[frame_key].shape[0] == batch_size
            ), f"{frame_key}'s shape is {batch[frame_key].shape}"
