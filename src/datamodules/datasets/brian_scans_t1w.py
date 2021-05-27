import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import json
from collections import Counter
from glob import glob
import os
from typing import List
from tqdm import tqdm
import nibabel as nib
import torch
from pathlib import Path


# class BrianScansT1w(Dataset):
#
#     def __init__(self, dataset_path, data_dir, transform=None):
#         my_file = Path(f"{data_dir}/T1w.pkl")
#         if not my_file.is_file():
#             data = {"scans": [], "labels":[], "gender":[], "age":[]}
#             paths = glob(f'{dataset_path}/*/anat/sub-*_T1w.nii.gz')
#             participants = pd.read_csv(f'{dataset_path}/participants.tsv', sep="\t")
#             for i, path in tqdm(enumerate(paths)):
#                 obj = nib.load(path)
#                 obj_data = obj.get_fdata()
#                 sub = os.path.split(os.path.dirname(os.path.dirname(path)))[1]
#                 row = participants.query(f"participant_id == '{sub}'")
#                 for j, scan in tqdm(enumerate(obj_data), leave=False):
#                     data["labels"].append(row.values[0][1])
#                     data["gender"].append(row.values[0][3])
#                     data["age"].append(row.values[0][2])
#                     data["scans"].append(scan)
#                 #     if j == 4:
#                 #         break
#                 # if i == 4:
#                 #     break
#             self.data = pd.DataFrame(data)
#             del data
#             del participants
#             self.data.to_pickle(f"{data_dir}/T1w.pkl")
#         else:
#             self.data = pd.read_pickle(f"{data_dir}/T1w.pkl")
#         self.transform = transform
#         self.classes = set(self.data.labels.values)
#         self.num_classes = len(self.classes)
#         self.encoder = dict(zip(self.classes, range(self.num_classes)))
#         self.decoder = dict((v, k) for k, v in self.encoder.items())
#
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         scan = np.uint8(np.expand_dims(self.data.iloc[index]["scans"], 2))
#         if self.transform:
#             scan = self.transform(scan)
#         return {
#             "scan": scan,
#             "label": self.encoder[self.data.iloc[index]["labels"]],
#             "age": self.data.iloc[index]["age"],
#             "gender": self.data.iloc[index]["gender"]
#         }

class BrianScansT1w(Dataset):

    def __init__(self, dataset_path, transform=None):
        self.paths = glob(f'{dataset_path}/*/anat/sub-*_T1w.nii.gz')
        self.paths = [x for x in self.paths if os.path.basename(x) != 'sub-50005_T1w.nii.gz']
        self.participants = pd.read_csv(f'{dataset_path}/participants.tsv', sep="\t")
        self.transform = transform
        self.num_channels = 176
        self.classes = set(self.participants.diagnosis.values)
        self.num_classes = len(self.classes)
        self.encoder = dict(zip(self.classes, range(self.num_classes)))
        self.decoder = dict((v, k) for k, v in self.encoder.items())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        obj = nib.load(path)
        obj_data = obj.get_fdata()
        sub = os.path.split(os.path.dirname(os.path.dirname(path)))[1]
        row = self.participants.query(f"participant_id == '{sub}'")
        label = self.encoder[row.values[0][1]]
        gender = row.values[0][3]
        age = row.values[0][2]
        scan = np.uint8(obj_data).transpose(1, 2, 0)

        if self.transform:
            scan = self.transform(scan)
        return {
            "scan": scan,
            "label": label,
            "age": age,
            "gender": gender
        }
