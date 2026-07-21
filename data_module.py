import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class GestureTrialDataset(Dataset):
    def __init__(self, trials):
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial = self.trials[idx]

        emg = trial["ns5_vector"]      # time x 32
        targets = trial["trainKin"]    # time x 7

        if not torch.is_tensor(emg):
            emg = torch.tensor(emg, dtype=torch.float32)
        else:
            emg = emg.float()

        if not torch.is_tensor(targets):
            targets = torch.tensor(targets, dtype=torch.float32)
        else:
            targets = targets.float()

        # model expects batch x channels x time
        emg = emg.T          # 32 x time
        targets = targets.T  # 7 x time

        return {
            "emg": emg,
            "targets": targets,
            "gesture": trial["gesture"],
            "trial_num": trial["trial_num"],
        }


class GestureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path,
        batch_size=4,
        num_workers=0,
        # stride = 24000
        # window_length = 24000
    ):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.stride = stride
        # self.window_length = window_length

    def setup(self, stage=None):
        dataset = torch.load(self.path, weights_only=False)

        self.train_dataset = GestureTrialDataset(dataset["train"])
        self.val_dataset = GestureTrialDataset(dataset["val"])
        self.test_dataset = GestureTrialDataset(dataset["test"])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=pad_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=pad_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=pad_collate,
        )


def pad_collate(batch):
    """
    Pads variable-length trials so they can batch together.

    emg:     batch x 32 x max_time
    targets: batch x 7 x max_time
    """

    max_len = max(item["emg"].shape[1] for item in batch)

    emg_batch = []
    target_batch = []

    gestures = []
    trial_nums = []

    for item in batch:
        emg = item["emg"]
        targets = item["targets"]

        pad_len = max_len - emg.shape[1]

        emg = torch.nn.functional.pad(emg, (0, pad_len))
        targets = torch.nn.functional.pad(targets, (0, pad_len))

        emg_batch.append(emg)
        target_batch.append(targets)

        gestures.append(item["gesture"])
        trial_nums.append(item["trial_num"])

    return {
        "emg": torch.stack(emg_batch),
        "targets": torch.stack(target_batch),
        "gesture": torch.tensor(gestures),
        "trial_num": torch.tensor(trial_nums),
    }


# In[ ]:




