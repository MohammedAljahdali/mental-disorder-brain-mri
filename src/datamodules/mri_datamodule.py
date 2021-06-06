from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from src.datamodules.datasets.brian_scans_t1w import BrianScansT1w
from src.utils.utils import calculate_mean
from sklearn.model_selection import train_test_split
from src.utils import utils
import numpy as np
import torch

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
        self.labels_counter = None

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
        # mean, std = calculate_mean(dataset, dataset.num_channels)
        self.setup_transforms()
        dataset = BrianScansT1w(self.dataset_dir, transform=self.test_transforms)
        self.labels_counter = dataset.labels_counter
        train_dataset_idx, val_dataset_idx = train_test_split(
            np.arange(len(dataset.labels)),
            train_size=0.6,
            shuffle=True,
            stratify=dataset.labels,
            random_state=1
        )
        val_dataset_idx, test_dataset_idx = train_test_split(
            val_dataset_idx,
            train_size=0.5,
            shuffle=True,
            stratify=np.array(dataset.labels)[val_dataset_idx],
            random_state=1
        )
        self.train_dataset = torch.utils.data.Subset(dataset, indices=train_dataset_idx)
        self.val_dataset = torch.utils.data.Subset(dataset, indices=val_dataset_idx)
        self.test_dataset = torch.utils.data.Subset(dataset, indices=test_dataset_idx)
        print(f"Length of train is {len(self.train_dataset)}")
        print(f"Length of val is {len(self.val_dataset)}")
        print(f"Length of test is {len(self.test_dataset)}")
        self.train_dataset.dataset.transform = self.train_transforms

    def setup_transforms(self):
        mean = [0.03555044, 0.03644094, 0.03768612, 0.03933044, 0.04138806, 0.04390845,
                0.04686559, 0.05040561, 0.05488866, 0.0598497, 0.06535633, 0.07114838,
                0.07741066, 0.08397495, 0.09015995, 0.09561275, 0.10028325, 0.10464429,
                0.10884795, 0.11272568, 0.1161376, 0.11909771, 0.1223311, 0.12567622,
                0.12900978, 0.13258312, 0.13632759, 0.14008128, 0.14384651, 0.14763705,
                0.15057223, 0.15340333, 0.15568345, 0.1582634, 0.16053551, 0.1628206,
                0.16502095, 0.16747784, 0.16994729, 0.17247222, 0.17502013, 0.17749757,
                0.18003827, 0.18260108, 0.18498457, 0.18724774, 0.18951172, 0.19167611,
                0.1937309, 0.1956074, 0.19740985, 0.19896994, 0.20041078, 0.20153163,
                0.20255902, 0.20353528, 0.20445888, 0.20516288, 0.20576458, 0.20624929,
                0.20666833, 0.20689151, 0.20727192, 0.20740478, 0.20758775, 0.20796604,
                0.20835329, 0.20873604, 0.20921561, 0.20986974, 0.21050925, 0.2111183,
                0.21174388, 0.21232274, 0.21283979, 0.21335694, 0.2137878, 0.21412231,
                0.2144567, 0.21474577, 0.21493113, 0.21508262, 0.21541261, 0.215912,
                0.21629999, 0.21669907, 0.21700149, 0.21684902, 0.21686486, 0.2172964,
                0.21768935, 0.21822283, 0.21863202, 0.21888335, 0.21921261, 0.21952136,
                0.21977312, 0.2198995, 0.21999373, 0.21971101, 0.21961603, 0.21914673,
                0.21864955, 0.21820801, 0.21774024, 0.21724851, 0.21664351, 0.21629235,
                0.21603627, 0.21577134, 0.21549865, 0.21532742, 0.21502935, 0.21478984,
                0.21451452, 0.21409711, 0.21365262, 0.21315192, 0.2125571, 0.21187787,
                0.21113619, 0.2102074, 0.20907159, 0.20776799, 0.20639179, 0.20474851,
                0.20270969, 0.20069734, 0.1985597, 0.19620288, 0.19405762, 0.1918659,
                0.18974683, 0.18762912, 0.18522535, 0.18316402, 0.18097264, 0.17867895,
                0.17629378, 0.17363734, 0.17105392, 0.16822493, 0.16516284, 0.16186706,
                0.15828171, 0.15451169, 0.15066763, 0.14670572, 0.14279159, 0.13876267,
                0.13483779, 0.1311717, 0.12752733, 0.12386699, 0.12012876, 0.11637033,
                0.11270375, 0.10875326, 0.10432421, 0.0990345, 0.0930741, 0.08673254,
                0.08030588, 0.07383918, 0.06732377, 0.06130533, 0.05597395, 0.05127211,
                0.0472601, 0.044025, 0.0412619, 0.03913275, 0.03758261, 0.03641299,
                0.03561411, 0.03522878, ]
        std = [0.06803005, 0.07072002, 0.07450564, 0.079443, 0.08571071, 0.09307344,
               0.1010425, 0.11023663, 0.12135826, 0.13244933, 0.14425235, 0.15571473,
               0.167254, 0.17845949, 0.18846179, 0.19648997, 0.2026066, 0.20810119,
               0.21302142, 0.21713814, 0.22042945, 0.22316771, 0.22639252, 0.22939548,
               0.23242819, 0.23559724, 0.23919037, 0.24283087, 0.24646596, 0.25012693,
               0.25309173, 0.25575394, 0.2579583, 0.26033725, 0.26237134, 0.26442081,
               0.26634439, 0.26843778, 0.27047218, 0.27246562, 0.27446751, 0.27630454,
               0.2781014, 0.27986302, 0.28141551, 0.28283968, 0.28426665, 0.28561201,
               0.28690088, 0.2881579, 0.28937461, 0.29030356, 0.29114825, 0.29168372,
               0.29214104, 0.29259635, 0.29293159, 0.29310336, 0.29320688, 0.29319938,
               0.29312642, 0.29286225, 0.29273992, 0.29236581, 0.29208713, 0.29193259,
               0.29184538, 0.29170964, 0.29168197, 0.29195404, 0.29216955, 0.29241765,
               0.2926225, 0.29273861, 0.29271646, 0.29263695, 0.29257242, 0.29238284,
               0.29219282, 0.29189, 0.29148037, 0.29095512, 0.29059541, 0.29028949,
               0.28966341, 0.28906776, 0.28847442, 0.28787701, 0.28777537, 0.28842476,
               0.28931729, 0.29051657, 0.29148136, 0.29224929, 0.29311225, 0.29390509,
               0.29464288, 0.29523965, 0.29577783, 0.29594713, 0.29623477, 0.2962341,
               0.29615402, 0.29608266, 0.29598131, 0.29581731, 0.2955038, 0.29542323,
               0.29540279, 0.29541899, 0.29547572, 0.29557613, 0.29565281, 0.29571751,
               0.29575938, 0.29571197, 0.2956512, 0.29566676, 0.29555589, 0.29548786,
               0.29536504, 0.29507944, 0.29461386, 0.29398311, 0.29336361, 0.29245323,
               0.29120878, 0.28997606, 0.28864067, 0.28718624, 0.28584174, 0.28434571,
               0.28283307, 0.28134204, 0.27959669, 0.27794219, 0.27620444, 0.27449423,
               0.27257897, 0.270358, 0.26810549, 0.26562898, 0.26304082, 0.26012738,
               0.25704775, 0.25382875, 0.25047722, 0.24689257, 0.24326167, 0.2395838,
               0.23606098, 0.2328734, 0.2299421, 0.22699105, 0.22396253, 0.22079888,
               0.21772, 0.21408227, 0.20911748, 0.20225117, 0.1937677, 0.18413421,
               0.17367109, 0.16243242, 0.15000106, 0.13787893, 0.12607822, 0.11458374,
               0.10421457, 0.09498407, 0.08646723, 0.07967396, 0.07455624, 0.07059492,
               0.06807684, 0.0670008, ]
        self.train_transforms = transforms.Compose([
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Pad((12, 12, 12, 12)),
                transforms.RandomAffine(degrees=40, shear=(1,4), translate=(0.4, 0.4), scale=(0.5, 1.25)),
                transforms.RandomPerspective(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=mean, std=std),
                transforms.Resize((64, 64)),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1)),
                # transforms.Lambda(lambda x: x.permute(1, 0, 2, 3))
            ])
        ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.Resize((64, 64)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1)),
            transforms.Lambda(lambda x: x.permute(1, 0, 2, 3))
        ])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
