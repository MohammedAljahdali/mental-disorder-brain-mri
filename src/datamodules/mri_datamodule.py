from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from src.datamodules.datasets.brian_scans_t1w import BrianScansT1w
from src.utils.utils import calculate_mean

from src.utils import utils

log = utils.get_logger(__name__)

class MRIDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        dataset_dir,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (0.7, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_transforms = None
        self.test_transforms = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    @property
    def num_classes(self) -> int:
        return 4

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        # BrianScansT1w(dataset_path=self.dataset_dir)
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset = BrianScansT1w(self.dataset_dir)
        log.info(f"Calculating mean and std of the dataset")
        mean, std = calculate_mean(dataset, dataset.num_channels)
        self.setup_transforms(mean, std)
        dataset = BrianScansT1w(self.dataset_dir, transform=self.test_transforms)
        train_length = int(len(dataset) * self.train_val_test_split[0])
        val_length = int((len(dataset) - train_length) * 0.5)
        test_length = len(dataset) - train_length - val_length
        self.data_train, self.data_val, self.data_test = random_split(
            dataset=dataset,
            lengths=(train_length, val_length, test_length),
        )
        print(f"Length of train is {len(self.data_train)}")
        print(f"Length of val is {len(self.data_val)}")
        print(f"Length of test is {len(self.data_test)}")
        self.data_train.dataset.transform = self.train_transforms

    def setup_transforms(self, mean, std):
        self.train_transforms = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.test_transforms = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )