from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from distutils.version import LooseVersion

RESIZE_AOI = 256
RESIZE_FINAL = 227
VERSION_GTE_0_12_0 = LooseVersion(tf.__version__) >= LooseVersion('0.12.0')

# Name change in TF v 0.12.0
if VERSION_GTE_0_12_0:
    standardize_image = tf.image.per_image_standardization
else:
    standardize_image = tf.image.per_image_whitening
       
def make_single_crop_batch(image_data):
    """Process a single image, each with a single-look
    Args:
    filenames: list of paths
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    """
    image = tf.image.resize_images(image_data, (RESIZE_AOI, RESIZE_AOI))

    crops = []
    h = int(image.shape[0])
    w = int(image.shape[1])
    hl = h - RESIZE_FINAL
    wl = w - RESIZE_FINAL

    crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
    crops.append(standardize_image(crop))
    crops.append(tf.image.flip_left_right(crop))

    image_batch = tf.stack(crops)
    return image_batch

def make_multi_crop_batch(image_data):
    """Process a single image file.
    Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    """
    # Read the image file.
    image = tf.image.resize_images(image_data, (RESIZE_AOI, RESIZE_AOI))

    crops = []
    h = int(image.shape[0])
    w = int(image.shape[1])
    hl = h - RESIZE_FINAL
    wl = w - RESIZE_FINAL

    crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
    crops.append(standardize_image(crop))
    crops.append(tf.image.flip_left_right(crop))

    corners = [ (0, 0), (0, wl), (hl, 0), (hl, wl)] #, (int(hl/2), int(wl/2))]
    for corner in corners:
        ch, cw = corner
        cropped = tf.image.crop_to_bounding_box(image, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
        crops.append(standardize_image(cropped))
        flipped = tf.image.flip_left_right(cropped)
        crops.append(standardize_image(flipped))

    image_batch = tf.stack(crops)
    return image_batch
