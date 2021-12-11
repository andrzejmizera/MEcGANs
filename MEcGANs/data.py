import os
import sys
import numpy as np
import cv2
import random
import glob
import gdal

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


class NIRRGB2RGBCLOUD_SYNTHETIC(BaseTrain):
    def __init__(self, dir_nir, dir_rgb, with_cloud_mask, dir_cloud_mask, dir_cloud, rgb_cloud_file, nir_cloud_file, nir_cloud_penetrability, imlist_nir, imlist_rgb,
        imlist_cloud_mask, *args, **kwargs):
        super().__init__()
        self.nir = read_imlist(dir_nir, imlist_nir)
        self.rgb = read_imlist(dir_rgb, imlist_rgb)
        #self.cloud_mask = list(glob.glob(os.path.join(dir_cloud_mask, '*.png')))
        self.withCM = with_cloud_mask
        #if self.withCM:
        self.cloud_mask = read_imlist(dir_cloud_mask, imlist_cloud_mask)
        self.size = kwargs.pop('size')
        self.augmentation = kwargs.pop('augmentation')
        #self.real_cloud_rgb_image = cv2.imread(os.path.join(dir_cloud, rgb_cloud_file),1).astype(np.float32)
        self.real_cloud_rgb_image = plt.imread(os.path.join(dir_cloud, rgb_cloud_file))*255.
        #self.real_cloud_nir_image = cv2.imread(os.path.join(dir_cloud, nir_cloud_file),0).astype(np.float32)
        self.real_cloud_nir_image = plt.imread(os.path.join(dir_cloud, nir_cloud_file))*255.
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

        clouded_rgb = (1. - alpha) * rgb + alpha * self.real_cloud_rgb_image[:, :, :3]
        clouded_rgb = np.clip(clouded_rgb, 0., 255.)

        nir_alpha = alpha[:, :, 0]
        clouded_nir = (1 - self.nir_cloud_penetrability * nir_alpha) * nir + self.nir_cloud_penetrability * nir_alpha * self.real_cloud_nir_image
        clouded_nir = np.clip(clouded_nir, 0., 255.)

        nirrgb = np.concatenate((clouded_nir[:, :, None], clouded_rgb), axis=2)
        if self.withCM:
            cloud = nir_alpha * self.real_cloud_nir_image
            rgbcloud = np.concatenate((rgb, cloud[:, :, None]), axis=2)
        else:
            rgbcloud = rgb

        nirrgb = nirrgb.transpose(2, 0, 1) / 127.5 - 1.
        rgbcloud = rgbcloud.transpose(2, 0, 1) / 127.5 - 1.

        nirrgb, rgbcloud = self.transform(nirrgb, rgbcloud)

        return nirrgb, rgbcloud


class NIRRGB2RGBCLOUD(BaseTrain):
    def __init__(self, dir_nir, dir_rgb, dir_rgb_cloud_free, with_cloud_mask, dir_cloud_mask, imlist_file, *args, **kwargs):
        super().__init__()
        self.nir = read_imlist(dir_nir, imlist_file)
        self.rgb = read_imlist(dir_rgb, imlist_file)
        self.rgb_cloud_free = read_imlist(dir_rgb_cloud_free, imlist_file)
        self.withCM = with_cloud_mask
        if self.withCM:
            self.cloud_mask = read_imlist(dir_cloud_mask, imlist_file)
        self.size = kwargs.pop('size')
        self.augmentation = kwargs.pop('augmentation')

    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        #nir = cv2.imread(self.nir[i], 0).astype(np.float32)
        #rgb = cv2.imread(self.rgb[i], 1).astype(np.float32)

        # cv2.imread does not work correctly on our dataset for NIR images.
        nir = plt.imread(self.nir[i]+'.png').astype(np.float32)*255.
        nir = cv2.normalize(nir, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255.
        rgb = plt.imread(self.rgb[i]+'.png').astype(np.float32)*255.
        target_rgb = plt.imread(self.rgb_cloud_free[i]+'.png').astype(np.float32)*255.
        if self.withCM:
            cloud_mask = plt.imread(self.cloud_mask[i]+'.png').astype(np.float32)*255.

        nirrgb = np.concatenate((nir[:,:,None], rgb), axis=2)
        if self.withCM:
            rgbcloud = np.concatenate((target_rgb, cloud_mask[:,:,None]), axis=2)
        else:
            rgbcloud = target_rgb

        nirrgb = nirrgb.transpose(2, 0, 1) / 127.5 - 1.
        rgbcloud = rgbcloud.transpose(2, 0, 1) / 127.5 - 1.

        nirrgb, rgbcloud = self.transform(nirrgb, rgbcloud)

        return nirrgb, rgbcloud


class NIRRGB2RGBCLOUD_TIF(BaseTrain):
    def __init__(self, dir_nirrgb, dir_rgb_cloud_free, with_cloud_mask, dir_cloud_mask, imlist_file, *args, **kwargs):
        super().__init__()
        self.nirrgb = read_imlist(dir_nirrgb, imlist_file)
        self.rgb_cloud_free = read_imlist(dir_rgb_cloud_free, imlist_file)
        self.withCM = with_cloud_mask
        if self.withCM:
            self.cloud_mask = read_imlist(dir_cloud_mask, imlist_file)
        self.size = kwargs.pop('size')
        self.augmentation = kwargs.pop('augmentation')

    def __len__(self):
        return len(self.nirrgb)

    def get_example(self, i):
        nirrgb = gdal.Open(self.nirrgb[i]+'.tif').ReadAsArray()[[7,3,2,1],:,:]
        # Clip outlier values as in the paper by Meraner et al.
        nirrgb = np.clip(nirrgb,0,10000)
        nirrgb = cv2.normalize(nirrgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255.
        target_rgb = gdal.Open(self.rgb_cloud_free[i]+'.tif').ReadAsArray()[[3,2,1],:,:]
        target_rgb = np.clip(target_rgb,0,10000)
        target_rgb = cv2.normalize(target_rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255.
        if self.withCM:
            cloud_mask = plt.imread(self.cloud_mask[i]+'.png').astype(np.float32)

        if self.withCM:
            rgbcloud = np.concatenate((target_rgb, cloud_mask[None, :, :]), axis=0)
        else:
            rgbcloud = target_rgb

        nirrgb = nirrgb / 127.5 - 1.
        rgbcloud = rgbcloud / 127.5 - 1.

        nirrgb, rgbcloud = self.transform(nirrgb, rgbcloud)

        return nirrgb, rgbcloud


class SARRGB2RGBCLOUD_TIF(BaseTrain):
    def __init__(self, dir_sar, dir_rgb, dir_rgb_cloud_free, with_cloud_mask, dir_cloud_mask, imlist_file, *args, **kwargs):
        super().__init__()
        self.sar = read_imlist(dir_sar, imlist_file)
        self.rgb = read_imlist(dir_rgb, imlist_file)
        self.rgb_cloud_free = read_imlist(dir_rgb_cloud_free, imlist_file)
        self.withCM = with_cloud_mask
        if self.withCM:
            self.cloud_mask = read_imlist(dir_cloud_mask, imlist_file)
        self.size = kwargs.pop('size')
        self.augmentation = kwargs.pop('augmentation')

    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        sar = np.transpose(gdal.Open(self.sar[i]+'.tif').ReadAsArray(), [1, 2, 0])
        sar_HH = np.clip(sar[:,:,0],-25.0,0)
        sar_HV = np.clip(sar[:,:,1],-32.5,0)
        sar = np.concatenate((sar_HH[:,:,None], sar_HV[:,:,None]), axis=2)
        sar = cv2.normalize(sar, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255.
        rgb = np.transpose(gdal.Open(self.rgb[i]+'.tif').ReadAsArray()[[3,2,1],:,:], [1, 2, 0])
        rgb = np.clip(rgb,0,10000)
        rgb = cv2.normalize(rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255. 
        target_rgb = np.transpose(gdal.Open(self.rgb_cloud_free[i]+'.tif').ReadAsArray()[[3,2,1],:,:], [1, 2, 0])
        target_rgb = np.clip(target_rgb,0,10000)
        target_rgb = cv2.normalize(target_rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255.
        if self.withCM:
            cloud_mask = plt.imread(self.cloud_mask[i]+'.png').astype(np.float32)

        sarrgb = np.concatenate((sar, rgb), axis=2)
        if self.withCM:
            rgbcloud = np.concatenate((target_rgb, cloud_mask[:, :, None]), axis=2)
        else:
            rgbcloud = target_rgb

        sarrgb = sarrgb.transpose(2, 0, 1) / 127.5 - 1.
        rgbcloud = rgbcloud.transpose(2, 0, 1) / 127.5 - 1.

        sarrgb, rgbcloud = self.transform(sarrgb, rgbcloud)

        return sarrgb, rgbcloud


class SARMS132RGBCLOUD(BaseTrain):
    def __init__(self, dir_sar, dir_rgb, dir_rgb_cloud_free, with_cloud_mask, dir_cloud_mask, imlist_file, *args, **kwargs):
        super().__init__()
        self.sar = read_imlist(dir_sar, imlist_file)
        self.rgb = read_imlist(dir_rgb, imlist_file)
        self.rgb_cloud_free = read_imlist(dir_rgb_cloud_free, imlist_file)
        self.withCM = with_cloud_mask
        if self.withCM:
            self.cloud_mask = read_imlist(dir_cloud_mask, imlist_file)
        self.size = kwargs.pop('size')
        self.augmentation = kwargs.pop('augmentation')

    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        sar = np.transpose(gdal.Open(self.sar[i]+'.tif').ReadAsArray(), [1, 2, 0])
        # Clip as in the paper of Meraner et al.
        sar_HH = np.clip(sar[:,:,0],-25.0,0)
        sar_HV = np.clip(sar[:,:,1],-32.5,0)
        # Scale to the range [-1,1]
        sar_HH = 2. * (sar_HH - sar_HH.min()) / (sar_HH.max() - sar_HH.min()) - 1.
        sar_HV = 2. * (sar_HV - sar_HV.min()) / (sar_HV.max() - sar_HV.min()) - 1.

        sar = np.concatenate((sar_HH[:,:,None], sar_HV[:,:,None]), axis=2)

        rgb = np.transpose(gdal.Open(self.rgb[i]+'.tif').ReadAsArray(), [1, 2, 0])
        # Clip as in the paper of Meraner et al.
        rgb = np.clip(rgb,0,10000)

        bgr = rgb[:,:,1:4]

        # Divide by 2000 as in Meraner et al. work
        rgb = rgb / 2000.

        rgb[:,:,1:4] = 2. * cv2.normalize(bgr, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) - 1.

        target_rgb = np.transpose(gdal.Open(self.rgb_cloud_free[i]+'.tif').ReadAsArray()[[3,2,1],:,:], [1, 2, 0])
        target_rgb = np.clip(target_rgb,0,10000)
        target_rgb = cv2.normalize(target_rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255.
        if self.withCM:
            cloud_mask = plt.imread(self.cloud_mask[i]+'.png').astype(np.float32)

        sarrgb = np.concatenate((sar, rgb), axis=2)
        if self.withCM:
            rgbcloud = np.concatenate((target_rgb, cloud_mask[:, :, None]), axis=2)
        else:
            rgbcloud = target_rgb

        sarrgb = sarrgb.transpose(2, 0, 1)
        rgbcloud = rgbcloud.transpose(2, 0, 1) / 127.5 - 1.

        sarrgb, rgbcloud = self.transform(sarrgb, rgbcloud)

        return sarrgb, rgbcloud


class SARRGB2RGBCLOUD(BaseTrain):
    def __init__(self, dir_sar, dir_rgb, dir_rgb_cloud_free, with_cloud_mask, dir_cloud_mask, imlist_file, *args, **kwargs):
        super().__init__()
        self.sar = read_imlist(dir_sar, imlist_file)
        self.rgb = read_imlist(dir_rgb, imlist_file)
        self.rgb_cloud_free = read_imlist(dir_rgb_cloud_free, imlist_file)
        self.withCM = with_cloud_mask
        if self.withCM:
            self.cloud_mask = read_imlist(dir_cloud_mask, imlist_file)
        self.size = kwargs.pop('size')
        self.augmentation = kwargs.pop('augmentation')

    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        #nir = cv2.imread(self.nir[i], 0).astype(np.float32)
        #rgb = cv2.imread(self.rgb[i], 1).astype(np.float32)

        # cv2.imread does not work correctly on our dataset for NIR images.
        sar = np.transpose(gdal.Open(self.sar[i]+'.tif').ReadAsArray(), [1, 2, 0])
        sar = cv2.normalize(sar, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255.
        rgb = plt.imread(self.rgb[i]+'.png').astype(np.float32)*255.
        target_rgb = plt.imread(self.rgb_cloud_free[i]+'.png').astype(np.float32)*255.
        if self.withCM:
            cloud_mask = plt.imread(self.cloud_mask[i]+'.png').astype(np.float32)*255.

        sarrgb = np.concatenate((sar, rgb), axis=2)
        if self.withCM:
            rgbcloud = np.concatenate((target_rgb, cloud_mask[:, :, None]), axis=2)
        else:
            rgbcloud = target_rgb

        sarrgb = sarrgb.transpose(2, 0, 1) / 127.5 - 1.
        rgbcloud = rgbcloud.transpose(2, 0, 1) / 127.5 - 1.

        sarrgb, rgbcloud = self.transform(sarrgb, rgbcloud)

        return sarrgb, rgbcloud

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
