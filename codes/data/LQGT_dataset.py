import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.utils.data as data

try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


class LQGTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.LR_paths, self.GT_paths = None, None
        self.LR_size, self.GT_size = opt["LR_size"], opt["GT_size"]

        # read image list from numpy files
        if opt["data_type"] == "numpy":
            self.LR_paths = util.get_numpy_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )  # LR list
            self.GT_paths = util.get_numpy_paths(
                opt["data_type"], opt["dataroot_GT"]
            )  # GT list
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.GT_paths, "Error: GT paths are empty."
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(
                self.GT_paths
                ), "GT and LR datasets have different number of images - {}, {}.".format(
                    len(self.LR_paths), len(self.GT_paths)
                )
        self.random_scale_list = [1]

    def __getitem__(self, index):

        GT_path, LR_path = None, None
        scale = self.opt["scale"] if self.opt["scale"] else 1
        GT_size = self.opt["GT_size"]
        LR_size = self.opt["LR_size"]

        # get GT image
        GT_path = self.GT_paths[index]
        resolution = None
        img_GT = util.read_numpy_file(GT_path)

        # get LR image
        if self.LR_paths:  # LR exist
            LR_path = self.LR_paths[index]
            resolution = None
            img_LR = util.read_numpy_file(LR_path)
            
        # 이미지가 흑백인 경우 차원 추가 (필요한 경우에만), 차원 확인 및 추가 로직을 여기로 이동
        if img_GT.ndim == 2:
            img_GT = np.expand_dims(img_GT, axis=2)
        if img_LR.ndim == 2:
            img_LR = np.expand_dims(img_LR, axis=2)
            
        if self.opt["phase"] == "train":
            H, W, C = img_LR.shape
            assert LR_size == GT_size // scale, "GT size does not match LR size"
        
            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            
        
        # 훈련 단계가 아닐 때 실행되어야 할 코드를 여기에 배치
        if self.opt["phase"] != "train" and LR_size is not None:
            H, W, C = img_LR.shape
            assert LR_size == GT_size // scale, "GT size does not match LR size"
        
            if LR_size < H and LR_size < W:
                # center crop
                rnd_h = H // 2 - LR_size // 2
                rnd_w = W // 2 - LR_size // 2
                img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[
                    rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :
                ]
        
        # 텐서 변환 부분 수정
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float() if img_GT.ndim == 3 else torch.from_numpy(
            np.ascontiguousarray(img_GT).reshape((1, img_GT.shape[0], img_GT.shape[1]))
        ).float()
        
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float() if img_LR.ndim == 3 else torch.from_numpy(
            np.ascontiguousarray(img_LR).reshape((1, img_LR.shape[0], img_LR.shape[1]))
        ).float()
 
        if LR_path is None:
            LR_path = GT_path

        return {"LQ": img_LR, "GT": img_GT, "LQ_path": LR_path, "GT_path": GT_path}

    def __len__(self):
        return len(self.GT_paths)