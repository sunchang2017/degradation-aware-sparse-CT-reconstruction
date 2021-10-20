import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
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


class _SparseTrainDataset(Dataset):
    def __init__(self, transform, dicom_root, sample_view=60, do_random_sample=False):
        self.transform = transforms.Compose(transform)
        names = os.listdir(dicom_root)
        names.sort()

        self.paths_dicom = [os.path.join(dicom_root, name) for name in names]
        self.total = len(self.paths_dicom)
        self.sample_view = sample_view
        self.do_random_sample = do_random_sample

    def __len__(self):
        return self.total

    def __getitem__(self, i):
        path_dicom = self.paths_dicom[i]
        split_name = os.path.splitext(path_dicom)[0]
        split_name = os.path.split(split_name)[1]
        img_gt = guiyi(read_dicom_img(path_dicom))
        if self.do_random_sample:
            sample_angle_list = [60, 120, 240]
            flag = np.random.randint(0, 3)
            sample_angle = sample_angle_list[flag]
        else:
            sample_angle = self.sample_view
        input_img_path = os.path.join('/data1G/sc/sparse_view/train/sparse_'+str(sample_angle),
                                      'img_input', split_name+'.png')
        with Image.open(input_img_path) as img_input:
            img_input = img_input.convert("L")
            img_input = np.array(img_input, dtype=np.float32)
            img_input = guiyi(img_input)
        return {"img_gt": self.transform(img_gt), \
                "img_input": self.transform(img_input), \
                "sample_angle": sample_angle,
                "split_name": split_name}


class SparseTrainDataset():
    def __init__(self, dicom_root, batch_size=1,
                 num_workers=0, transform=[transforms.ToTensor()], sample_view=60,
                 do_random_sample=False):
        self.batchsize = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.dicom_root = dicom_root
        self.sample_view = sample_view
        self.do_random_sample = do_random_sample
        self.train_dataset = _SparseTrainDataset(transform=self.transform, dicom_root=self.dicom_root,
                                                 sample_view=self.sample_view,
                                                 do_random_sample=self.do_random_sample)

    def build_datasets(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batchsize,
                                      shuffle=True, num_workers=self.num_workers)
        return train_dataloader


class _SparseValDataset(Dataset):
    def __init__(self, transform, gt_root, input_root):
        self.transform = transforms.Compose(transform)
        img_gt_root = os.path.join(gt_root, 'image')
        img_input_root = os.path.join(input_root, 'img_input')
        names = os.listdir(img_gt_root)
        names.sort()
        self.paths_img_gt = [os.path.join(img_gt_root, name) for name in names]
        self.paths_img_input = [os.path.join(img_input_root, name) for name in names]
        self.total = len(names)

    def __len__(self):
        return self.total

    def __getitem__(self, i):
        path_img_gt = self.paths_img_gt[i]
        path_img_input = self.paths_img_input[i]
        split_name = os.path.splitext(path_img_input)[0]
        split_name = os.path.split(split_name)[1]
        with Image.open(path_img_gt) as img_gt:
            img_gt = img_gt.convert("L")
            img_gt = np.array(img_gt, dtype=np.float32)
        with Image.open(path_img_input) as img_input:
            img_input = img_input.convert("L")
            img_input = np.array(img_input, dtype=np.float32)
        return {"img_gt": self.transform(guiyi(img_gt)),
                "img_input": self.transform(guiyi(img_input)),
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







