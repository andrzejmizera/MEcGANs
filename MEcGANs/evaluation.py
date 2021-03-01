import os
import numpy as np
import cv2

import chainer
from chainer import Variable


def get_batch(test_iter, xp):
    batch = test_iter.next()
    batchsize = len(batch)

    x = []
    for j in range(batchsize):
        x.append(np.asarray(batch[j][0]).astype("f"))
    x = Variable(xp.asarray(x))

    return x, batchsize


def save_images(input_image, output_image, results_dir, current_n):
    for i, (x, y) in enumerate(zip(input_image, output_image)):
        x = x.transpose(1, 2, 0).astype(np.uint8)
        y = y.transpose(1, 2, 0).astype(np.uint8)
        if x.shape[2] == 1:
            cv2.imwrite(os.path.join(results_dir, '{:03d}_input_nir.png'.format(current_n + i)), x)
        elif x.shape[2] == 3:
            cv2.imwrite(os.path.join(results_dir, '{:03d}_input_rgb.png'.format(current_n + i)), x)
        elif x.shape[2] == 4:
            cv2.imwrite(os.path.join(results_dir, '{:03d}_input_nir.png'.format(current_n + i)), x[:, :, 0])
            cv2.imwrite(os.path.join(results_dir, '{:03d}_input_rgb.png'.format(current_n + i)), x[:, :, 1:])
        else:
            raise NotImplementedError
        if y.shape[2] == 3:
            cv2.imwrite(os.path.join(results_dir, '{:03d}_output_rgb.png'.format(current_n + i)), y)
        elif y.shape[2] == 4:
            cv2.imwrite(os.path.join(results_dir, '{:03d}_output_rgb.png'.format(current_n + i)), y[:, :, :3])
            cv2.imwrite(os.path.join(results_dir, '{:03d}_output_cloud.png'.format(current_n + i)), y[:, :, 3])
        else:
            raise NotImplementedError


def out_image(test_iter, gen, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        test_iter.reset()
        xp = gen.xp
        results_dir = os.path.join(dst, 'test_{:03d}'.format(trainer.updater.iteration))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        n = 0
        while True:
            x, batchsize = get_batch(test_iter, xp)

            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                out = gen(x)
                out = np.clip(out.array.get() * 127.5 + 127.5, 0., 255.)
                x = x.array.get() * 127.5 + 127.5

                save_images(x, out, results_dir, current_n=n)
                n += len(out)

            if test_iter.is_new_epoch:
                test_iter.reset()
                break

    return make_image
