import sys
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import random

CHANNELS = 3
HEIGHT = 224
WIDTH = 224


# Copied from Pytorch src

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


# def is_image_file(filename):
#     """Checks if a file is an allowed image extension.
#     Args:
#         filename (string): path to a file
#     Returns:
#         bool: True if the filename ends with a known image extension
#     """
#     return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    filenames = []
    labels = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    # item = (path, class_to_idx[target])
                    filenames.append(path)
                    labels.append(int(class_to_idx[target]))

    return filenames, labels

def _find_classes(dir):
    """
    Finds the class folders in a dataset.
    Args:
        dir (string): Root directory path.
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def _return_dataset(dir):
    classes, class_to_idx = _find_classes(dir)
    filenames, labels = make_dataset(dir, class_to_idx, IMG_EXTENSIONS)
    return filenames, labels


def preprocess_image(filename, label):
    # IMAGENET_MEAN = [123.68, 116.779, 103.939] # rgb format
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    img = Image.open(filename).convert('RGB')

    # resize of the image (setting lowest dimension to 256px)
    if img.size[0] < img.size[1]:
        h = int(float(256 * img.size[1]) / img.size[0])
        img = img.resize((256, h), Image.ANTIALIAS)
    else:
        w = int(float(256 * img.size[0]) / img.size[1])
        img = img.resize((w, 256), Image.ANTIALIAS)

    # random 244x224 patch
    x = random.randint(0, img.size[0] - 224)
    y = random.randint(0, img.size[1] - 224)
    img_cropped = img.crop((x, y, x + 224, y + 224))

    cropped_im_array = np.array(img_cropped, dtype=np.float32)

    cropped_im_array/=255.0

    for i in range(3):
        cropped_im_array[:,:,i] -= IMAGENET_MEAN[i]
        cropped_im_array[:,:,i] /= IMAGENET_STD[i]

    #for i in range(3):
    #   mean = np.mean(img_c1_np[:,:,i])
    #   stddev = np.std(img_c1_np[:,:,i])
    #   img_c1_np[:,:,i] -= mean
    #   img_c1_np[:,:,i] /= stddev

    # tf_image = tf.convert_to_tensor(cropped_im_array, dtype=tf.float32)

    cropped_im_array = np.clip(cropped_im_array, 0.0, 1.0)

    return cropped_im_array, label


def get_imagenet_generator(data_dir):
    """
    Gets a generator function that will be used for the input_fn
    Parameters
    ----------
    data_dir: str
        Path to where the imagenet data resides

    Returns
    -------
    generator_fn: callable
        A generator function that will yield feature dict and label
    """
    classes, class_to_idx = _find_classes(data_dir)
    filenames, labels = make_dataset(data_dir, class_to_idx, IMG_EXTENSIONS)
    dataset_len = len(labels)

    def generator():
        for i in range(dataset_len):
            yield (preprocess_image(filenames[i], labels[i]))

    return generator


def get_input_fn(data_dir, num_epochs, batch_size, shuffle):
    """
    This will return input_fn from which batches of data can be obtained.
    Parameters
    ----------
    data_dir: str
        Path to where the imagenet data resides
    num_epochs: int
        Number of data epochs
    is_training: bool
        Whether to read the training or the test portion of the data
    batch_size: int
        Batch size
    shuffle: bool
        Whether to shuffle the data or not

    Returns
    -------
    input_fn: callable
        The input function which returns a batch of images and labels
        tensors, of shape (batch size, HEIGTH, WIDTH, CHANNELS) and
        (batch size), respectively.
    """

    gen = get_imagenet_generator(data_dir)              # Tensor slices?
    ds = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=(tf.float32, tf.int64),
        output_shapes=(tf.TensorShape([HEIGHT, WIDTH, CHANNELS]),
                       tf.TensorShape([]))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    # ds = ds.map(preprocess_image, num_parallel_calls=4)
    ds = ds.repeat(count=num_epochs)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(10 * batch_size)




    def input_fn():
        ds_iter = ds.make_one_shot_iterator()
        images, labels = ds_iter.get_next()
        return images, labels

    return input_fn


if __name__ == '__main__':

    train_input_fn = get_input_fn(
        data_dir='/nfs1/datasets/imagenet_nfs1/train',
        num_epochs=1,
        batch_size=64,
        shuffle=True)

    val_input_fn = get_input_fn(
        data_dir='/nfs1/datasets/imagenet_nfs1/val',
        num_epochs=1,
        batch_size=64,
        shuffle=False)

    train_im, train_lbl = train_input_fn()
    val_im, val_lbl = val_input_fn()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # tf.train.start_queue_runners(sess)
        im, lbl = sess.run((train_im, train_lbl))
        print(im.shape, lbl.shape)

        im, lbl = sess.run((val_im, val_lbl))
        print(im.shape, lbl.shape)



# (tensorflow_gpuenv) ani0075:imagenet$ python data_utils.py
# 2018-12-07 15:30:31.880948: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# 2018-12-07 15:30:41.923444: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:97] Filling up shuffle buffer (this may take a while): 1034 of 10000
# 2018-12-07 15:30:51.918386: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:97] Filling up shuffle buffer (this may take a while): 2356 of 10000
# 2018-12-07 15:31:01.917699: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:97] Filling up shuffle buffer (this may take a while): 3599 of 10000
# 2018-12-07 15:31:11.938712: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:97] Filling up shuffle buffer (this may take a while): 4804 of 10000
# 2018-12-07 15:31:21.919184: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:97] Filling up shuffle buffer (this may take a while): 6047 of 10000
# 2018-12-07 15:31:31.921378: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:97] Filling up shuffle buffer (this may take a while): 7158 of 10000
# 2018-12-07 15:31:41.934950: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:97] Filling up shuffle buffer (this may take a while): 8242 of 10000
# 2018-12-07 15:31:51.978135: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:97] Filling up shuffle buffer (this may take a while): 9282 of 10000
# 2018-12-07 15:31:58.627555: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:135] Shuffle buffer filled.
# (64, 224, 224, 3) (64,)
# (64, 224, 224, 3) (64,)