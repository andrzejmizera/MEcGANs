import os
import sys
import numpy as np
import cv2
import random
import glob

import chainer

from matplotlib import pyplot as plt

def read_imlist(root_dir, txt_imlist):
    with open(txt_imlist, 'r') as f:
        ret = [os.path.join(root_dir, path.strip()) for path in f.readlines()]
    return ret


def train_test_dataset(train_class_name, args_train, test_class_name, args_test):
    mod_name = os.path.splitext(os.path.basename(__file__))[0]
    mod_path = os.path.dirname(__file__)
    sys.path.insert(0, mod_path)
    train_class = getattr(__import__(mod_name), train_class_name)
    test_class = getattr(__import__(mod_name), test_class_name)
    train = train_class(**args_train)
    test = test_class(**args_test)

    return train, test


class TestNIRRGB(chainer.dataset.DatasetMixin):
    def __init__(self, dir_nir, dir_rgb, imlist_nir, imlist_rgb):
        super().__init__()
        self.nir = read_imlist(dir_nir, imlist_nir)
        self.rgb = read_imlist(dir_rgb, imlist_rgb)

    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        nir = cv2.imread(self.nir[i], 0).astype(np.float32)
        rgb = cv2.imread(self.rgb[i], 1).astype(np.float32)

        nirrgb = np.concatenate((nir[:, :, None], rgb), axis=2)
        nirrgb = nirrgb.transpose(2, 0, 1) / 127.5 - 1.

        return nirrgb,


class TestNIR(chainer.dataset.DatasetMixin):
    def __init__(self, dir_nir, imlist_nir):
        super().__init__()
        self.nir = read_imlist(dir_nir, imlist_nir)

    def __len__(self):
        return len(self.nir)

    def get_example(self, i):
        nir = cv2.imread(self.nir[i], 0).astype(np.float32)
        nir = nir[None, :, :] / 127.5 - 1.

        return nir,


class TestRGB(chainer.dataset.DatasetMixin):
    def __init__(self, dir_rgb, imlist_rgb):
        super().__init__()
        self.rgb = read_imlist(dir_rgb, imlist_rgb)

    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        rgb = cv2.imread(self.rgb[i], 1).astype(np.float32)
        rgb = rgb.transpose(2, 0, 1) / 127.5 - 1.

        return rgb,


class BaseTrain(chainer.dataset.DatasetMixin):
    def __init__(self):
        super().__init__()

    def transform(self, x, y):
        c, h, w = x.shape
        if self.augmentation:
            top = random.randint(0, h - self.size - 1)
            left = random.randint(0, w - self.size - 1)
            if random.randint(0, 1):
                x = x[:, :, ::-1]
                y = y[:, :, ::-1]
        else:
            top = (h - self.size) // 2
            left = (w - self.size) // 2
        bottom = top + self.size
        right = left + self.size

        x = x[:, top:bottom, left:right]
        y = y[:, top:bottom, left:right]

        return x, y


class NIRRGB2RGBCLOUD(BaseTrain):
    def __init__(self, dir_nir, dir_rgb, dir_cloud_mask, dir_cloud, rgb_cloud_file, nir_cloud_file, nir_cloud_penetrability, imlist_nir, imlist_rgb,
        imlist_cloud_mask, *args, **kwargs):
        super().__init__()
        self.nir = read_imlist(dir_nir, imlist_nir)
        self.rgb = read_imlist(dir_rgb, imlist_rgb)
        #self.cloud_mask = list(glob.glob(os.path.join(dir_cloud_mask, '*.png')))
        self.cloud_mask = read_imlist(dir_cloud_mask, imlist_cloud_mask)
        self.size = kwargs.pop('size')
        self.augmentation = kwargs.pop('augmentation')
        #self.real_cloud_rgb_image = cv2.imread(os.path.join(dir_cloud, rgb_cloud_file),1).astype(np.float32)
        self.real_cloud_rgb_image = plt.imread(os.path.join(dir_cloud, rgb_cloud_file))*255.
        #self.real_cloud_nir_image = cv2.imread(os.path.join(dir_cloud, nir_cloud_file),0).astype(np.float32)
        self.real_cloud_nir_image =plt.imread(os.path.join(dir_cloud, nir_cloud_file))*255.
        self.nir_cloud_penetrability = nir_cloud_penetrability


    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        #nir = cv2.imread(self.nir[i], 0).astype(np.float32)
        #rgb = cv2.imread(self.rgb[i], 1).astype(np.float32)
        
        # cv2.imread does not work correctly on our dataset for NIR images.
        # NIR image values need to be rescaled to [0., 255.] for proper alpha blending with the cloud image.
        nir = plt.imread(self.nir[i]).astype(np.float32)
        nir = cv2.normalize(nir, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255.
        rgb = plt.imread(self.rgb[i]).astype(np.float32)*255.

        alpha = plt.imread(self.cloud_mask[i]).astype(np.float32)
        alpha = np.broadcast_to(alpha[:, :, None], alpha.shape + (3,))
        
        clouded_rgb = (1. - alpha) * rgb + alpha * self.real_cloud_rgb_image
        clouded_rgb = np.clip(clouded_rgb, 0., 255.)
        
        nir_alpha = alpha[:, :, 0]
        clouded_nir = (1 - self.nir_cloud_penetrability * nir_alpha) * nir + self.nir_cloud_penetrability * nir_alpha * self.real_cloud_nir_image
        clouded_nir = np.clip(clouded_nir, 0., 255.)
        
        nirrgb = np.concatenate((clouded_nir[:, :, None], clouded_rgb), axis=2)
        cloud = nir_alpha * self.real_cloud_nir_image
        rgbcloud = np.concatenate((rgb, cloud[:, :, None]), axis=2)

        nirrgb = nirrgb.transpose(2, 0, 1) / 127.5 - 1.
        rgbcloud = rgbcloud.transpose(2, 0, 1) / 127.5 - 1.

        nirrgb, rgbcloud = self.transform(nirrgb, rgbcloud)

        return nirrgb, rgbcloud


class RGB2RGBCLOUD(BaseTrain):
    def __init__(self, dir_rgb, dir_cloud, imlist_rgb, *args, **kwargs):
        super().__init__()
        self.rgb = read_imlist(dir_rgb, imlist_rgb)
        self.cloud = list(glob.glob(os.path.join(dir_cloud, '*.png')))
        self.size = kwargs.pop('size')
        self.augmentation = kwargs.pop('augmentation')

    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        rgb = cv2.imread(self.rgb[i], 1).astype(np.float32)
        cloud = cv2.imread(random.choice(self.cloud), -1).astype(np.float32)

        alpha = cloud[:, :, 3] / 255.
        alpha = np.broadcast_to(alpha[:, :, None], alpha.shape + (3,))
        clouded_rgb = (1. - alpha) * rgb + alpha * cloud[:, :, :3]
        clouded_rgb = np.clip(clouded_rgb, 0., 255.)

        cloud = cloud[:, :, 3]
        rgbcloud = np.concatenate((rgb, cloud[:, :, None]), axis=2)

        rgb = clouded_rgb.transpose(2, 0, 1) / 127.5 - 1.
        rgbcloud = rgbcloud.transpose(2, 0, 1) / 127.5 - 1.

        rgb, rgbcloud = self.transform(rgb, rgbcloud)

        return rgb, rgbcloud


class NIR2RGB(BaseTrain):
    def __init__(self, dir_nir, dir_rgb, imlist_nir, imlist_rgb, *args, **kwargs):
        super().__init__()
        self.nir = read_imlist(dir_nir, imlist_nir)
        self.rgb = read_imlist(dir_rgb, imlist_rgb)
        self.size = kwargs.pop('size')
        self.augmentation = kwargs.pop('augmentation')

    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        nir = cv2.imread(self.nir[i], 0).astype(np.float32)
        rgb = cv2.imread(self.rgb[i], 1).astype(np.float32)

        nir = nir[None, :, :] / 127.5 - 1.
        rgb = rgb.transpose(2, 0, 1) / 127.5 - 1.

        nir, rgb = self.transform(nir, rgb)

        return nir, rgb
