# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import cv2
from cryptography.fernet import Fernet
from torch_utils.misc import get_key, read_key
import pickle

import random
import pickle
import torch
from torchvision import transforms
from albumentations import KeypointParams, RandomBrightnessContrast, RGBShift, HorizontalFlip, Compose
from training.transforms import ScaledCropNearMask
from pathlib import Path


try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution=None,        # Image resolution
        key=None,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self._resolution = resolution
        self._key = key

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            data = f.read()
            if self._key is not None:
                data = Fernet(self._key).decrypt(data)
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(data)
            else:
                image = np.asarray(bytearray(data), dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        if self._resolution is not None:
            interpolation = cv2.INTER_AREA if self._resolution < image.shape[0] else cv2.INTER_LANCZOS4
            if image.shape[0] != self._resolution or image.shape[1] != self._resolution:
                image = cv2.resize(image, (self._resolution, self._resolution), interpolation=interpolation)
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels


class ImageFolderDatasetDescr(ImageFolderDataset):
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)

    def _load_raw_labels(self):
        fname = 'descriptors.pkl'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = pickle.load(f)
        if labels is None:
            return None
        labels = np.array(labels)
        return labels

#----------------------------------------------------------------------------

class ContoursDataset(torch.utils.data.Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution=None,        # Image resolution
        name=None,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        super(ContoursDataset, self).__init__()
        self._path = Path(path)
        self._resolution = resolution
        self._name = name

        self._color_transform = Compose([
            RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
            RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
        ], p=1.0)
        additional_targets = {'soft_mask': 'image', 'contours_mask': 'image', 'obj_mask': 'mask'}
        interpolation = cv2.INTER_LINEAR
        self._transform = Compose([
            ScaledCropNearMask(height=self._resolution, width=self._resolution, always_apply=True, interpolation=interpolation),
            HorizontalFlip(),
        ], p=1.0, additional_targets=additional_targets)

        self._dataset_samples = self._load_samples(str(self._path / 'contours.json'))
        if len(self._dataset_samples) == 0:
            raise IOError('No image files found in the specified path')

        self._name = self._path.stem if name is None else name

    @property
    def image_shape(self):
        return (3, self._resolution, self._resolution)

    @property
    def label_shape(self):
        return (0,)

    @property
    def label_dim(self):
        return 0

    @property
    def has_labels(self):
        return False

    @property
    def num_channels(self):
        return 1

    @property
    def map_channels(self):
        return 3

    @property
    def resolution(self):
        return self._resolution

    @property
    def name(self):
        return self._name

    def _load_samples(self, json_file):
        with open(json_file, 'r') as fp:
            data = json.load(fp)

        samples = []
        for base_name in data:
            imdata = data[base_name]
            imname, maskname, contours_raw = imdata
            if len(contours_raw) == 0:
                print(f'Warning: image with name={base_name} has no contours!')
                continue
            contours = []
            for contour in contours_raw:
                is_positive = contour[0]
                coords = contour[1]
                contours.append((is_positive, coords))
            samples.append((imname, maskname, contours))

        return samples

    def _get_filled_mask(self, contours, mask_shape):

        pos_mask = np.zeros(mask_shape, dtype=np.uint8)
        neg_mask = np.zeros(mask_shape, dtype=np.uint8)

        for is_positive, coords in contours:
            filled_mask = np.zeros(mask_shape, dtype=np.uint8)
            if len(coords) > 2:
                pts = [np.array(coords)[:, ::-1].astype(np.int32)]
                cv2.fillPoly(filled_mask, pts=pts, color=255)
            if is_positive:
                pos_mask = np.maximum(pos_mask, filled_mask)
            else:
                neg_mask = np.maximum(neg_mask, filled_mask)

        mask = np.stack((pos_mask, neg_mask), axis=2)
        maskf = mask.astype(np.float32) / 255.

        return maskf

    def __getitem__(self, idx):
        imname, maskname, _ = self._dataset_samples[idx]

        image = cv2.imread(str(self._path / imname), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_mask = cv2.imread(str(self._path / maskname), 0)
        mask = np.zeros_like(raw_mask)
        mask[raw_mask > 128] = 1
        obj_mask = np.array(mask, dtype=np.int32)
        mask = mask.astype(np.float32)

        if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            obj_mask = cv2.resize(obj_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        image = self._color_transform(image=image)['image']

        aug_out = self._transform(image=image, soft_mask=mask, obj_mask=obj_mask)
        image = aug_out['image']
        mask = aug_out['soft_mask']

        image = 2 * (image.transpose(2, 0, 1) / 255. - 0.5)
        mask = (2.0 * mask - 1)
        mask = mask[np.newaxis, :, :]

        label = np.zeros([1, 0], dtype=np.float32)

        return image, mask, label[0]

    def get_label(self, idx):
        label = np.zeros([1, 0], dtype=np.float32)
        return label[0]

    def __len__(self):
        return len(self._dataset_samples)


class OpenImagesDataset(torch.utils.data.Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution=None,        # Image resolution
        name=None,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        super(OpenImagesDataset, self).__init__()
        self._path = Path(path)
        self._resolution = resolution
        self._name = name

        self._color_transform = Compose([
            RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
            RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
        ], p=1.0)
        additional_targets = {'soft_mask': 'image', 'obj_mask': 'mask'}
        interpolation = cv2.INTER_LINEAR
        self._transform = Compose([
            ScaledCropNearMask(height=self._resolution, width=self._resolution, always_apply=True, interpolation=interpolation),
            HorizontalFlip(),
        ], p=1.0, additional_targets=additional_targets)

        test_dataset_samples = self._load_samples(self._path, split='test')
        val_dataset_samples = self._load_samples(self._path, split='val')
        self._dataset_samples = test_dataset_samples + val_dataset_samples
        if len(self._dataset_samples) == 0:
            raise IOError('No image files found in the specified path')

        self._name = self._path.stem if name is None else name

    @property
    def image_shape(self):
        return (3, self._resolution, self._resolution)

    @property
    def label_shape(self):
        return (0,)

    @property
    def label_dim(self):
        return 0

    @property
    def has_labels(self):
        return False

    @property
    def num_channels(self):
        return 1

    @property
    def map_channels(self):
        return 3

    @property
    def resolution(self):
        return self._resolution

    @property
    def name(self):
        return self._name

    def _load_samples(self, base_dir, split):

        split_path = base_dir / split
        images_dirname = f'{split}/images'
        masks_dirname = f'{split}/masks'

        anno_path = str(split_path / f'{split}-annotations-object-segmentation.csv')
        if os.path.exists(anno_path):
            with open(anno_path, 'r') as f:
                data = f.read().splitlines()
        else:
            raise RuntimeError(f'Can\'t find annotations at {anno_path}')

        image_id_to_masks = {}
        for line in data[1:]:
            parts = line.split(',')
            if '.png' in parts[0]:
                mask_name = parts[0]
                image_id = parts[1]
            else:
                mask_name = parts[1]
                image_id = parts[2]
            if image_id not in image_id_to_masks:
                image_id_to_masks[image_id] = []
            image_id_to_masks[image_id].append(mask_name)

        image_id_to_masks = image_id_to_masks
        dataset_samples = list(image_id_to_masks.keys())

        dataset_samples_n = []
        for image_id in dataset_samples:
            masknames = image_id_to_masks[image_id]
            imname = f'{images_dirname}/{image_id}.jpg'
            for maskname in masknames:
                maskname = f'{masks_dirname}/{maskname}'
                dataset_samples_n.append((imname, maskname))
        dataset_samples = dataset_samples_n
        return dataset_samples

    def __getitem__(self, idx):
        imname, maskname = self._dataset_samples[idx]

        image_path = str(self._path / imname)
        mask_path = str(self._path / maskname)

        image = cv2.imread(image_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)
        mask = mask > 0
        obj_mask = np.array(mask, dtype=np.int32)
        mask = mask.astype(np.float32)

        if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            obj_mask = cv2.resize(obj_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        image = self._color_transform(image=image)['image']

        aug_out = self._transform(image=image, soft_mask=mask, obj_mask=obj_mask)
        image = aug_out['image']
        mask = aug_out['soft_mask']

        image = 2 * (image.transpose(2, 0, 1) / 255. - 0.5)
        mask = (2.0 * mask - 1)
        mask = mask[np.newaxis, :, :]

        label = np.zeros([1, 0], dtype=np.float32)

        return image, mask, label[0]

    def get_label(self, idx):
        label = np.zeros([1, 0], dtype=np.float32)
        return label[0]

    def __len__(self):
        return len(self._dataset_samples)