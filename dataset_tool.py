# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import functools
import io
import json
import os
import pickle
import sys
import tarfile
import gzip
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, List

import click
import numpy as np
import PIL.Image
from tqdm import tqdm
import cv2
import random
from cryptography.fernet import Fernet
import itertools
from pathlib import Path
from os.path import splitext

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None and b > 0:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in ('.jpg', '.jpeg', '.png', '.bmp', '.heic') # type: ignore

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int]):

    if os.path.isfile(os.path.join(source_dir, 'images_list.txt')):
        with open(os.path.join(source_dir, 'images_list.txt')) as fp:
            input_images = fp.read().splitlines()
        input_images = [os.path.join(source_dir, name) for name in input_images]
    else:
        input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]
    random.shuffle(input_images)
    # Load labels.
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            # img = np.array(PIL.Image.open(fname))
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield dict(img=img, label=labels.get(arch_fname))
            if idx >= max_idx-1:
                break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]

        # Load labels.
        labels = {}
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                labels = json.load(file)['labels']
                if labels is not None:
                    labels = { x[0]: x[1] for x in labels }
                else:
                    labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    # img = PIL.Image.open(file) # type: ignore
                    # img = np.array(img)
                    img = np.asarray(bytearray(file.read()), dtype='uint8')
                    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = cv2.imread(file, 1)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                yield dict(img=img, label=labels.get(fname))
                if idx >= max_idx-1:
                    break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_lmdb(lmdb_dir: str, *, max_images: Optional[int]):
    import cv2  # pip install opencv-python
    import lmdb  # pip install lmdb # pylint: disable=import-error

    with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
        max_idx = maybe_min(txn.stat()['entries'], max_images)

    def iterate_images():
        with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.array(PIL.Image.open(io.BytesIO(value)))
                    yield dict(img=img, label=None)
                    if idx >= max_idx-1:
                        break
                except:
                    print(sys.exc_info()[1])

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_cifar10(tarball: str, *, max_images: Optional[int]):
    images = []
    labels = []

    with tarfile.open(tarball, 'r:gz') as tar:
        for batch in range(1, 6):
            member = tar.getmember(f'cifar-10-batches-py/data_batch_{batch}')
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding='latin1')
            images.append(data['data'].reshape(-1, 3, 32, 32))
            labels.append(data['labels'])

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = images.transpose([0, 2, 3, 1]) # NCHW -> NHWC
    assert images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_mnist(images_gz: str, *, max_images: Optional[int]):
    labels_gz = images_gz.replace('-images-idx3-ubyte.gz', '-labels-idx1-ubyte.gz')
    assert labels_gz != images_gz
    images = []
    labels = []

    with gzip.open(images_gz, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    with gzip.open(labels_gz, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int],
    resize_filter: str
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    # resample = { 'box': PIL.Image.BOX, 'lanczos': PIL.Image.LANCZOS }[resize_filter]
    resample = { 'box': cv2.INTER_NEAREST, 'lanczos': cv2.INTER_LANCZOS4 }[resize_filter]

    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        # img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        # img = img.resize((ww, hh), resample)
        img = cv2.resize(img, (ww, hh), interpolation=resample)
        # return np.array(img)
        return img

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        # img = PIL.Image.fromarray(img, 'RGB')
        # img = img.resize((width, height), resample)
        img = cv2.resize(img, (width, height), interpolation=resample)
        # return np.array(img)
        return img

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        # img = PIL.Image.fromarray(img, 'RGB')
        # img = img.resize((width, height), resample)
        # img = np.array(img)
        img = cv2.resize(img, (width, height), interpolation=resample)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'none':
        return lambda x:x
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error ('must specify --width and --height when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error ('must specify --width and --height when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_datasets(sources, *, max_images_list: Optional[List[int]]):
    nums_samples = []
    image_iters = []
    for max_images, source in zip(max_images_list, sources):
        if os.path.isdir(source):
            if source.rstrip('/').endswith('_lmdb'):
                num_samples, image_iter = open_lmdb(source, max_images=max_images)
                image_iters.append(image_iter)
                nums_samples.append(num_samples)
            else:
                num_samples, image_iter = open_image_folder(source, max_images=max_images)
                image_iters.append(image_iter)
                nums_samples.append(num_samples)
        elif os.path.isfile(source):
            if os.path.basename(source) == 'cifar-10-python.tar.gz':
                num_samples, image_iter = open_cifar10(source, max_images=max_images)
                image_iters.append(image_iter)
                nums_samples.append(num_samples)
            elif os.path.basename(source) == 'train-images-idx3-ubyte.gz':
                num_samples, image_iter = open_mnist(source, max_images=max_images)
                image_iters.append(image_iter)
                nums_samples.append(num_samples)
            elif file_ext(source) == 'zip':
                num_samples, image_iter = open_image_zip(source, max_images=max_images)
                image_iters.append(image_iter)
                nums_samples.append(num_samples)
            else:
                assert False, 'unknown archive type'
        else:
            error(f'Missing input file or directory: {source}')

    return sum(nums_samples), itertools.chain(*image_iters)

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--max-images', help='Output only up to `max-images` images', type=str, default=None)
@click.option('--resize-filter', help='Filter to use when resizing images for output resolution', type=click.Choice(['box', 'lanczos']), default='lanczos', show_default=True)
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide', 'none']))
@click.option('--width', help='Output width', type=int)
@click.option('--height', help='Output height', type=int)
@click.option('--use-jpg', help='Use JPG', is_flag=True)
@click.option('--encrypt', help='Encrypt data', is_flag=True)
def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    max_images: Optional[str],
    transform: Optional[str],
    resize_filter: str,
    width: Optional[int],
    height: Optional[int],
    use_jpg: bool,
    encrypt: bool
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source *_lmdb/                    Load LSUN dataset
    --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
    --source train-images-idx3-ubyte.gz Load MNIST dataset
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Recursively load all images from dataset.zip

    The output dataset format can be either an image folder or a zip archive.
    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    Images within the dataset archive will be stored as uncompressed PNG.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --width and --height options.  Output resolution will be either the original
    input resolution (if --width/--height was not specified) or the one specified with
    --width/height.

    Use the --transform=center-crop or --transform=center-crop-wide options to apply a
    center crop transform on the input image.  These options should be used with the
    --width and --height options.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop-wide --width 512 --height=384
    """

    # PIL.Image.init() # type: ignore

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')

    sources = source.split(',')
    if isinstance(max_images, str):
        parts = max_images.split(',')
        if len(parts) == 1 and len(sources) > 1:
            parts = parts * len(sources)
        max_images = [int(v) for v in parts]
    else:
        max_images = [max_images] * len(sources)
    assert len(max_images) == len(sources)
    num_files, input_iter = open_datasets(sources, max_images_list=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)

    transform_image = make_transform(transform, width, height, resize_filter)

    dataset_attrs = None
    key = None
    if encrypt:
        key = Fernet.generate_key()
        print("Key:", key.decode())

        if not os.path.isdir(dest):
            dest = os.path.dirname(dest)

        with open(os.path.join(dest, 'key.txt'), 'w') as fp:
            fp.write(key.decode())

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        ext = '.jpg' if use_jpg else '.png'
        archive_fname = f'{idx_str[:5]}/img{idx_str}{ext}'

        # Apply crop and resize.
        img = transform_image(image['img'])

        # Transform may drop images.
        if img is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.

        # channels = img.shape[2] if img.ndim == 3 else 1
        # cur_image_attrs = {
        #     'width': img.shape[1],
        #     'height': img.shape[0],
        #     'channels': channels
        # }
        # if dataset_attrs is None:
        #     dataset_attrs = cur_image_attrs
        #     width = dataset_attrs['width']
        #     height = dataset_attrs['height']
        #     if width != height:
        #         error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
        #     if dataset_attrs['channels'] not in [1, 3]:
        #         error('Input images must be stored as RGB or grayscale')
        #     if width != 2 ** int(np.floor(np.log2(width))):
        #         error('Image width/height after scale and crop are required to be power-of-two')
        # elif dataset_attrs != cur_image_attrs:
        #     err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
        #     error(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed PNG.
        # img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB' }[channels])
        # image_bits = io.BytesIO()

        # image_bytes = io.BytesIO(cv2.imencode(ext, img)[1].tostring())
        # image_bytes = image_bytes.getbuffer()
        image_bytes = cv2.imencode(ext, img)[1].tostring()
        if encrypt:
            image_bytes = Fernet(key).encrypt(image_bytes)
        # img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bytes)
        labels.append([archive_fname, image['label']] if image['label'] is not None else None)

    metadata = {
        'labels': labels if all(x is not None for x in labels) else None
    }
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()


def load_contours(json_file):
    with open(json_file, 'r') as fp:
        data = json.load(fp)

    samples = {}
    for base_name in data:
        imdata = data[base_name]
        base_name = splitext(base_name)[0]
        labels = {}
        for obj_id in imdata:
            contours_raw = imdata[obj_id]
            if len(contours_raw) == 0:
                print(f'Warning: image with name={base_name} has no contours!')
                continue
            contours = []
            for contour in contours_raw:
                is_positive = contour[0]
                coords = contour[1]
                contours.append((is_positive, coords))
            obj_id = int(obj_id)
            assert len(contours) > 0
            labels[obj_id] = contours
        if len(labels) > 0:
            samples[base_name] = labels

    return samples


def load_berkeley_dataset(base_dir, contours_file='contours.json', images_dir_name='images', masks_dir_name='masks'):
    base_dir = Path(base_dir)
    images_dir = base_dir / images_dir_name
    masks_dir = base_dir / masks_dir_name

    dataset_samples = [x.name for x in sorted(images_dir.glob('*.*'))]
    masks_paths = {x.stem: x.name for x in masks_dir.glob('*.*')}
    dataset_samples = [(imname, masks_paths[Path(imname).stem]) for imname in dataset_samples]

    contours_path = base_dir / contours_file
    contours_data = load_contours(contours_path)
    dataset_samples_n = []
    for imname, masksname in dataset_samples:
        base_name = Path(imname).stem
        if base_name not in contours_data:
            continue
        image_contours = contours_data[base_name]
        obj_keys = list(image_contours.keys())
        assert len(obj_keys) == 1
        contours = image_contours[obj_keys[0]]
        dataset_samples_n.append((imname, masksname, contours))
    dataset_samples = dataset_samples_n
    dataset_samples = [(str(images_dir / imname), str(masks_dir / maskname), contours)
                       for imname, maskname, contours in dataset_samples]

    def load_sample_func(item):
        image_path = item[0]
        mask_path = item[1]
        contours = item[2]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        instances_mask[instances_mask == 128] = -1
        instances_mask[instances_mask > 128] = 1

        return image, instances_mask, contours

    return dataset_samples, load_sample_func


def load_grabcut_dataset(base_dir, contours_file='contours.json'):
    return load_berkeley_dataset(base_dir, contours_file=contours_file,
                                 images_dir_name='data_GT', masks_dir_name='boundary_GT')

def load_davis_dataset(base_dir, contours_file='contours.json'):
    dataset_samples, _ = load_berkeley_dataset(base_dir, contours_file=contours_file,
                                            images_dir_name='img', masks_dir_name='gt')

    def load_sample_func(item):
        image_path = item[0]
        mask_path = item[1]
        contours = item[2]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = np.max(cv2.imread(mask_path).astype(np.int32), axis=2)
        instances_mask[instances_mask > 0] = 1

        return image, instances_mask, contours

    return dataset_samples, load_sample_func


def load_openimages_dataset(base_dir, split='test', contours_file='contours.json', exclude_file=None):
    base_dir = Path(base_dir)
    split_path = base_dir / split
    images_path = split_path / 'images'
    masks_path = split_path / 'masks'
    exclude_file = split_path / exclude_file if exclude_file is not None else None

    if split == 'test' and exclude_file is None:
        print('Exclude file is not specified for OpenImages(test), trying to use `exclude.txt`')
        exclude_file = split_path / 'exclude.txt'
        if not exclude_file.exists():
            msg = 'Failed to find `exclude.txt` file, RefinedOI images is not excluded from OpenImages(test)'
            print(msg)

    anno_path = str(split_path / f'{split}-annotations-object-segmentation.csv')
    if os.path.exists(anno_path):
        with open(anno_path, 'r') as f:
            data = f.read().splitlines()
    else:
        raise RuntimeError(f'Can\'t find annotations at {anno_path}')

    images_exclude = None
    if exclude_file is not None:
        with open(str(exclude_file), 'r') as fp:
            images_exclude = fp.read().splitlines()
        images_exclude = set(images_exclude)

    image_id_to_masks = {}
    excluded = 0
    for line in data[1:]:
        parts = line.split(',')
        if '.png' in parts[0]:
            mask_name = parts[0]
            image_id = parts[1]
        else:
            mask_name = parts[1]
            image_id = parts[2]
        if images_exclude is not None and image_id in images_exclude:
            excluded += 1
            continue
        if image_id not in image_id_to_masks:
            image_id_to_masks[image_id] = []
        image_id_to_masks[image_id].append(mask_name)

    if excluded > 0:
        print(f'Number of excluded masks: {excluded}')

    image_id_to_masks = image_id_to_masks
    dataset_samples = list(image_id_to_masks.keys())

    contours_path = split_path / contours_file
    contours_data = load_contours(contours_path)
    dataset_samples_n = []
    for image_id in dataset_samples:
        if image_id not in contours_data:
            continue
        dataset_samples_n.append(image_id)
    dataset_samples = dataset_samples_n
    dataset_samples_n = []
    for image_id in dataset_samples:
        masknames = image_id_to_masks[image_id]
        objects_contours = contours_data[image_id]
        imname = str(images_path / f'{image_id}.jpg')
        only_objects = objects_contours.keys()
        masks_and_contours = [(masknames[obj_index], objects_contours[obj_index]) for obj_index in only_objects]
        for maskname, contours in masks_and_contours:
            maskname = str(masks_path / maskname)
            dataset_samples_n.append((imname, maskname, contours))
    dataset_samples = dataset_samples_n

    def load_sample_func(item):
        image_path = item[0]
        mask_path = item[1]
        contours = item[2]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)
        object_mask = mask > 0
        instances_mask = np.array(object_mask, dtype=np.int32)

        return image, instances_mask, contours

    return dataset_samples, load_sample_func

#----------------------------------------------------------------------------
def prepare_contours_dataset():

    base_dir = '/media/denemmy/hdd/data'
    base_dir = '/media/hdd/data'

    datasets = [(load_berkeley_dataset, f'{base_dir}/interactive_segmentation/RefinedOI', 'contours.json'),
                (load_davis_dataset, f'{base_dir}/interactive_segmentation/DAVIS', 'contours.json'),
                (load_grabcut_dataset, f'{base_dir}/interactive_segmentation/GrabCut', 'contours.json'),
                (load_berkeley_dataset, f'{base_dir}/interactive_segmentation/Berkeley', 'contours.json'),
                (load_openimages_dataset, f'{base_dir}/interactive_segmentation/openimages', 'val', 'contours_val.json'),
                (load_openimages_dataset, f'{base_dir}/interactive_segmentation/openimages', 'test', 'OpenImages_Test_v2.json')]

    # datasets = [(load_openimages_dataset, '/media/denemmy/hdd/data/open_images', 'val', 'contours_val.json')]

    output_dir = Path(f'{base_dir}/interactive_segmentation/contours_dataset')
    output_dir.mkdir(exist_ok=True, parents=True)

    img_dirname = 'images'
    mask_dirname = 'masks'

    output_imgs_dir = output_dir / img_dirname
    output_mask_dir = output_dir / mask_dirname

    output_imgs_dir.mkdir(exist_ok=True, parents=True)
    output_mask_dir.mkdir(exist_ok=True, parents=True)

    all_samples = []
    output_samples = {}
    index = 0
    for items in datasets:
        load_op = items[0]
        base_dir = items[1]
        args = items[1:]
        samples, load_samples_func = load_op(*args)
        print(f'{base_dir}: {len(samples)}')
        images_unique = {}
        for item in samples:
            image, instances_mask, contours = load_samples_func(item)
            img_path = item[0]
            imname = f'{img_dirname}/{index:04d}.jpg'
            maskname = f'{mask_dirname}/{index:04d}.png'
            store_img = True
            if img_path in images_unique:
                imname = images_unique[img_path]
                store_img = False
            else:
                images_unique[img_path] = imname

            mask = np.array(instances_mask)
            mask[instances_mask < 0] = 128
            mask[instances_mask > 0] = 255
            mask = mask.astype(np.uint8)

            if store_img:
                cv2.imwrite(str(output_dir / imname), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output_dir / maskname), mask)

            output_samples[f'{index:04d}'] = (imname, maskname, contours)

            index += 1

    with open(str(output_dir / 'contours.json'), 'w') as fp:
        json.dump(output_samples, fp, indent=4)


if __name__ == "__main__":
    prepare_contours_dataset() # pylint: disable=no-value-for-parameter
