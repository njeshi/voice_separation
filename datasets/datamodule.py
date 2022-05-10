import os
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from datasets import LibriSpeech


class LibriSpeechDatamodule(pl.LightningDataModule):
    def __init__(self, hp, root, batch_size, download):
        super().__init__()

        self.hp = hp
        self.data_dir = root
        self.batch_size = batch_size
        self.download = download

    def prepare_data(self):
        # Download and prepare
        self.train_ds = LibriSpeech(self.hp, self.data_dir, train=True, download=self.download)
        self.test_ds = LibriSpeech(self.hp, self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            full_dataset = LibriSpeech(self.hp, self.data_dir, train=True)

            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.libri_train, self.libri_val = random_split(full_dataset,
                                                            [train_size, val_size],
                                                            generator=torch.Generator())

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.libri_test = LibriSpeech(self.hp, self.data_dir, train=False)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        collate_fn = self.train_ds.train_collate_fn

        return DataLoader(dataset=self.libri_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=os.cpu_count(),
                          collate_fn=collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        collate_fn = self.test_ds.test_collate_fn

        return DataLoader(dataset=self.libri_test,
                          collate_fn=collate_fn,
                          batch_size=1,
                          shuffle=False,
                          num_workers=os.cpu_count())

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        collate_fn = self.train_ds.train_collate_fn

        return DataLoader(dataset=self.libri_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=collate_fn,
                          num_workers=os.cpu_count(),
                          drop_last=False)
