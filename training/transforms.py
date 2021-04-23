import random

import cv2
import numpy as np
from albumentations import DualTransform
from albumentations.augmentations import functional as F


class ScaledCropNearMask(DualTransform):
    """Random scale, crop and resize to target shape if mask is non-empty, else make random crop.

    Args:
        height (int): vertical size of crop in pixels
        width (int): horizontal size of crop in pixels
        ignore_values (list of int): values to ignore in mask, `0` values are always ignored
            (e.g. if background value is 5 set `ignore_values=[5]` to ignore)
        ignore_channels (list of int): channels to ignore in mask
            (e.g. if background is a first channel set `ignore_channels=[0]` to ignore)
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, obj_max_scale=0.9, obj_min_scale=0.1, padding_scale=0.01,
                 ignore_values=None, ignore_channels=None, always_apply=False, p=1.0,
                 interpolation=cv2.INTER_LINEAR, desired_image_max_scale=2.5, desired_image_min_scale=0.4):
        super(ScaledCropNearMask, self).__init__(always_apply, p)

        if ignore_values is not None and not isinstance(ignore_values, list):
            raise ValueError("Expected `ignore_values` of type `list`, got `{}`".format(type(ignore_values)))
        if ignore_channels is not None and not isinstance(ignore_channels, list):
            raise ValueError("Expected `ignore_channels` of type `list`, got `{}`".format(type(ignore_channels)))

        self.height = height
        self.width = width
        self.ignore_values = ignore_values
        self.ignore_channels = ignore_channels
        self.obj_max_scale = obj_max_scale
        self.obj_min_scale = obj_min_scale
        self.desired_image_max_scale = desired_image_max_scale
        self.desired_image_min_scale = desired_image_min_scale
        self.padding_scale = padding_scale
        self.interpolation = interpolation
        assert self.padding_scale < 1

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, interpolation=cv2.INTER_LINEAR, **params):

        img_height, img_width = img.shape[:2]

        top_pad = max(0, -y_min)
        left_pad = max(0, -x_min)
        bottom_pad = max(0, y_max - img_height)
        right_pad = max(0, x_max - img_width)
        img = F.pad_with_params(img, top_pad, bottom_pad, left_pad, right_pad, border_mode=cv2.BORDER_CONSTANT, value=0)
        if left_pad > 0:
            x_min = x_min + left_pad
            x_max = x_max + left_pad
        if top_pad > 0:
            y_min = y_min + top_pad
            y_max = y_max + top_pad
        img = F.crop(img, x_min, y_min, x_max, y_max)
        img = F.resize(img, self.height, self.width, interpolation=interpolation)

        return img

    def apply_to_bbox(self, bbox, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        raise NotImplementedError

    def apply_to_keypoint(self, keypoint, x_min=0, x_max=0, y_min=0, y_max=0, **params):

        crop_width = x_max - x_min
        crop_height = y_max - y_min

        crop_coords = (x_min, y_min, x_max, y_max)

        keypoint = F.crop_keypoint_by_coords(keypoint, crop_coords, crop_height=crop_height, crop_width=crop_width,
                                             rows=params["rows"], cols=params["cols"])
        scale_x = self.width / crop_width
        scale_y = self.height / crop_height
        keypoint = F.keypoint_scale(keypoint, scale_x, scale_y)

        return keypoint

    @property
    def targets_as_params(self):
        return ["obj_mask"]

    def get_params_dependent_on_targets(self, params):
        mask = params["obj_mask"]
        mask_height, mask_width = mask.shape[:2]

        if self.ignore_values is not None:
            ignore_values_np = np.array(self.ignore_values)
            mask = np.where(np.isin(mask, ignore_values_np), 0, mask)

        if mask.ndim == 3 and self.ignore_channels is not None:
            target_channels = np.array([ch for ch in range(mask.shape[-1]) if ch not in self.ignore_channels])
            mask = np.take(mask, target_channels, axis=-1)

        if self.width >= mask_width:
            x_min = int(0.5 * (mask_width - self.width))
        else:
            x_min = random.randint(0, mask_width - self.width)
        if self.height >= mask_height:
            y_min = int(0.5 * (mask_height - self.height))
        else:
            y_min = random.randint(0, mask_height - self.height)
        crop_width = self.width
        crop_height = self.height

        if mask.sum() > 0:
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            values = self.find_params(mask)
            if values is not None:
                x_min, y_min, crop_width, crop_height = values

        x_max = x_min + crop_width
        y_max = y_min + crop_height

        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def find_params(self, mask):

        # find bbox of the mask
        non_zero_yx = np.argwhere(mask)
        y_min = np.min(non_zero_yx[:,0])
        y_max = np.max(non_zero_yx[:,0])
        x_min = np.min(non_zero_yx[:,1])
        x_max = np.max(non_zero_yx[:,1])

        mask_w = x_max - x_min + 1
        mask_h = y_max - y_min + 1

        # the mask is too small
        if mask_w < 16 or mask_h < 16:
            return None

        min_side = min(mask_h, mask_w)

        # allow padding
        padding_size = int(self.padding_scale * min_side)

        min_w = max(0, mask_w - padding_size)
        min_h = max(0, mask_h - padding_size)
        # the mask is too small
        if min_w < 4 or min_h < 4:
            return None

        # detect min/max scale of the image based on min/max object size
        crop_wh_ratio = float(self.width) / self.height
        min_wh_ratio = float(min_w) / min_h
        if crop_wh_ratio < min_wh_ratio:
            crop_min_width = min_w
            crop_min_height = int(min_w / crop_wh_ratio + 0.5)
        else:
            crop_min_height = min_h
            crop_min_width = int(min_h * crop_wh_ratio + 0.5)

        current_scale = self.width / float(crop_min_width)

        mult_min_image_scale = current_scale * self.obj_min_scale
        mult_max_image_scale = current_scale * self.obj_max_scale

        # clip [min, max] scale if it's outside of the desired min or max image scale
        mult_scale = None
        if mult_max_image_scale > self.desired_image_max_scale:
            if mult_min_image_scale > self.desired_image_max_scale:
                mult_scale = mult_min_image_scale
            else:
                mult_max_image_scale = self.desired_image_max_scale

        if mult_min_image_scale < self.desired_image_min_scale:
            if mult_max_image_scale < self.desired_image_min_scale:
                mult_scale = mult_max_image_scale
            else:
                mult_min_image_scale = self.desired_image_min_scale

        if mult_scale is None:
            if mult_min_image_scale > mult_max_image_scale - 1e-3:
                mult_scale = mult_max_image_scale
            else:
                mult_scale = np.random.uniform(mult_min_image_scale, mult_max_image_scale)

        # crop size
        crop_width = int(self.width / mult_scale + 0.5)
        crop_height = int(self.height / mult_scale + 0.5)

        # crop position
        x_start_max = x_min + padding_size
        x_start_min = x_max - padding_size - crop_width
        if x_start_min > x_start_max - 1e-3:
            x_start = 0.5 * (x_start_min + x_start_max)
        else:
            x_start = np.random.randint(x_start_min, x_start_max)

        y_start_max = y_min + padding_size
        y_start_min = y_max - padding_size - crop_height
        if y_start_min > y_start_max - 1e-3:
            y_start = 0.5 * (y_start_min + y_start_max)
        else:
            y_start = np.random.randint(y_start_min, y_start_max)

        return x_start, y_start, crop_width, crop_height

    def get_transform_init_args_names(self):
        return ("height", "width", "ignore_values", "ignore_channels")