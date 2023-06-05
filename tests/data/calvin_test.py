import pytest

from torch.utils.data import DataLoader
import torch

from nlgoals.data.calvin.legacy.dataset import CALVIN


@pytest.fixture
def calvin():
    data_dir = "data/calvin/calvin_debug_dataset/"
    split = "training"
    frame_keys = ["rgb_static", "actions", "rel_actions"]
    num_frames = 5
    dataset = CALVIN(data_dir, split, num_frames, frame_keys)
    return dataset


@pytest.fixture
def calvin_allframes():
    data_dir = "data/calvin/calvin_debug_dataset/"
    split = "training"
    frame_keys = None
    num_frames = 5
    dataset = CALVIN(data_dir, split, num_frames, frame_keys)
    return dataset


def test_length(calvin):
    # we know that the debug dset has 5124 samples
    assert len(calvin) == 9


def test_getitem(calvin):
    item = calvin[0]
    assert isinstance(item, dict)

    expected_keys = set(["lang_ann", "task"] + calvin.frame_keys)
    assert (
        set(item.keys()) == expected_keys
    ), f"Key mismatch: item: {item.keys()}, expected: {expected_keys}"


@pytest.mark.parametrize(
    "dataset", [pytest.lazy_fixture("calvin"), pytest.lazy_fixture("calvin_allframes")]
)
def test_dataloading(dataset):
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

    for batch in dataloader:
        assert isinstance(batch, dict)

        expected_keys = set(["lang_ann", "task"] + dataset.frame_keys)
        assert (
            set(batch.keys()) == expected_keys
        ), f"Key mismatch: batch: {batch.keys()}, expected: {expected_keys}"

        for frame_key in dataset.frame_keys:
            print(frame_key, batch[frame_key].shape)
            assert frame_key in batch
            assert isinstance(batch[frame_key], torch.Tensor)
            assert batch[frame_key].shape[1] == dataset.num_frames
            assert (
                batch[frame_key].shape[0] == batch_size
            ), f"{frame_key}'s shape is {batch[frame_key].shape}"
