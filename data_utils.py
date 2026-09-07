"""Validated IDX loading and preprocessing for 28 x 28 grayscale images."""
import gzip
import struct
from pathlib import Path
import numpy as np


def load_mnist(path, kind='train'):
    path = Path(path)
    with gzip.open(path / f'{kind}-labels-idx1-ubyte.gz', 'rb') as stream:
        labels_bytes = stream.read()
    with gzip.open(path / f'{kind}-images-idx3-ubyte.gz', 'rb') as stream:
        images_bytes = stream.read()
    if len(labels_bytes) < 8 or len(images_bytes) < 16:
        raise ValueError('Truncated IDX header')
    label_magic, label_count = struct.unpack('>II', labels_bytes[:8])
    image_magic, image_count, rows, columns = struct.unpack('>IIII', images_bytes[:16])
    if label_magic != 2049 or image_magic != 2051 or (rows, columns) != (28, 28):
        raise ValueError('Expected uint8 IDX labels and 28 x 28 grayscale images')
    if image_count == 0 or label_count != image_count:
        raise ValueError('Image and label counts must match and be nonzero')
    if len(labels_bytes) != 8 + label_count or len(images_bytes) != 16 + image_count * 784:
        raise ValueError('IDX payload size does not match its header')
    labels = np.frombuffer(labels_bytes, dtype=np.uint8, offset=8)
    if np.any(labels > 9):
        raise ValueError('Expected class labels from 0 to 9')
    return np.frombuffer(images_bytes, dtype=np.uint8, offset=16).reshape(image_count, 784), labels


def prepare_images(images):
    images = np.asarray(images)
    if images.ndim not in (2, 3, 4) or images.shape[0] == 0 or images.shape[1:] not in ((784,), (28,28), (28,28,1)):
        raise ValueError('Expected nonempty 28 x 28 grayscale images')
    if not np.isfinite(images).all() or np.any(images < 0) or np.any(images > 255):
        raise ValueError('Expected finite pixel intensities from 0 to 255')
    return images.astype(np.float32).reshape(-1, 28, 28, 1) / np.float32(255)
