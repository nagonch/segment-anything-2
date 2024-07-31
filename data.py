import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import math
import h5py


class HCIOldDataset(Dataset):
    def __init__(self, data_path="HCI_dataset_old"):
        self.data_path = data_path
        self.scene_to_path = {}
        self.scenes = [
            "horses",
            "papillon",
            "stillLife",
            "buddha",
        ]
        for scene in self.scenes:
            self.scene_to_path[scene] = f"{data_path}/{scene}"

    def get_scene(self, name):
        scene = h5py.File(f"{self.scene_to_path[name]}/lf.h5", "r")
        LF = np.array(scene["LF"])
        return LF

    def get_labels(self, name):
        labels = np.array(
            h5py.File(f"{self.scene_to_path[name]}/labels.h5", "r")["GT_LABELS"]
        )
        return labels

    def get_depth(self, name):
        scene = h5py.File(f"{self.scene_to_path[name]}/lf.h5", "r")
        gt_depth = np.array(scene["GT_DEPTH"])
        return gt_depth

    def get_disparity(self, name, eps=1e-9):
        scene = h5py.File(f"{self.scene_to_path[name]}/lf.h5", "r")
        gt_depth = np.array(scene["GT_DEPTH"])
        s_size, t_size, u_size, v_size = gt_depth.shape
        dH = scene.attrs["dH"][0]
        f = scene.attrs["focalLength"][0]
        shift = scene.attrs["shift"][0]
        disparity = np.zeros((s_size, t_size, u_size, v_size, 2))
        central_ind = s_size // 2
        for s in range(s_size):
            for t in range(t_size):
                disparity[s, t, :, :, 0] = (dH * (central_ind - s)) * f / (
                    gt_depth[s, t] + eps
                ) - shift * (central_ind - s)
                disparity[s, t, :, :, 1] = (dH * (central_ind - t)) * f / (
                    gt_depth[s, t] + eps
                ) - shift * (central_ind - t)
        return disparity

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]
        LF = self.get_scene(scene_name)
        labels = self.get_labels(scene_name)
        disparity = self.get_disparity(scene_name)
        return LF, labels, disparity


if __name__ == "__main__":
    dataset = HCIOldDataset()
    LF, _, _ = dataset[0]
    print(LF.shape)
