import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

import pydicom
import cv2

def read_dicom_img(dicom_path, resize_pixel=512):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)
    img[img == -2000] = 0
    if img.shape[0] != 512 or img.shape[1] != 512:
        img = cv2.resize(img, (resize_pixel, resize_pixel), interpolation=cv2.INTER_CUBIC)

    RescaleSlope = ds.RescaleSlope
    RescaleIntercept = ds.RescaleIntercept
    CT_img = img * RescaleSlope + RescaleIntercept
    # CT_img translate to u
    u_water = 0.02
    u_img = CT_img / 1000 * u_water + u_water
    return u_img


def guiyi(x):
    x_max = x.max()
    x_min = x.min()
    x = (x - x_min) / (x_max - x_min)
    return x



class _SparseValDataset(Dataset):
    def __init__(self, transform, gt_root, input_root):
        self.transform = transforms.Compose(transform)
        img_gt_root = os.path.join(gt_root, 'image')
        img_input_root = os.path.join(input_root, 'img_input')
        # res_pic_root = os.path.join(input_root, 'res_pic')
        SE_result_root = os.path.join(input_root, 'SE_result')
        names = os.listdir(img_gt_root)
        names.sort()

        self.paths_img_gt = [os.path.join(img_gt_root, name) for name in names]
        self.paths_img_input = [os.path.join(img_input_root, name) for name in names]
        # self.paths_res_pic = [os.path.join(res_pic_root, name) for name in names]
        self.paths_SE_result = [os.path.join(SE_result_root, name) for name in names]
        self.total = len(names)

    def __len__(self):
        return self.total

    def __getitem__(self, i):
        path_img_gt = self.paths_img_gt[i]
        path_img_input = self.paths_img_input[i]
        # path_res_pic = self.paths_res_pic[i]
        path_SE_result = self.paths_SE_result[i]

        split_name = os.path.splitext(path_img_input)[0]
        split_name = os.path.split(split_name)[1]

        with Image.open(path_img_gt) as img_gt:
            img_gt = img_gt.convert("L")
            img_gt = np.array(img_gt, dtype=np.float32)
        with Image.open(path_img_input) as img_input:
            img_input = img_input.convert("L")
            img_input = np.array(img_input, dtype=np.float32)
        # with Image.open(path_res_pic) as res_pic:
        #     res_pic = res_pic.convert("L")
        #     res_pic = np.array(res_pic, dtype=np.float32)
        with Image.open(path_SE_result) as SE_result:
            SE_result = SE_result.convert("L")
            SE_result = np.array(SE_result, dtype=np.float32)

        return {"SE_result": self.transform(guiyi(SE_result)),
                "img_input": self.transform(guiyi(img_input)),
                # "res_pic":self.transform(guiyi(res_pic)),
                "img_gt": self.transform(img_gt),
                "split_name":split_name}


class SparseValDataset():
    def __init__(self, gt_root, input_root, batch_size=32,
                 num_workers=0,
                 transform=[transforms.ToTensor()]):
        self.input_root = input_root
        self.gt_root = gt_root
        self.batchsize = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.valDataset = _SparseValDataset(transform=self.transform, gt_root=self.gt_root, \
                                            input_root=self.input_root)

    def build_datasets(self):
        valid_dataloader = DataLoader(self.valDataset, batch_size=self.batchsize,
                                      shuffle=True, num_workers=self.num_workers)
        return valid_dataloader
