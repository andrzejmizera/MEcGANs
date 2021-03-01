import os
import numpy as np
import argparse
import yaml

import chainer

import source.yaml_utils as yaml_utils
from evaluation import get_batch, save_images


def load_gen(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    return gen


def test(args):
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    if args.dir_nir is not None:
        config.dataset['args']['args_test']['dir_nir'] = args.dir_nir
        config.dataset['args']['args_test']['imlist_nir'] = args.imlist_nir
    if args.dir_rgb is not None:
        config.dataset['args']['args_test']['dir_rgb'] = args.dir_rgb
        config.dataset['args']['args_test']['imlist_rgb'] = args.imlist_rgb
    chainer.cuda.get_device_from_id(0).use()
    gen = load_gen(config)
    chainer.serializers.load_npz(args.gen_model, gen)
    gen.to_gpu()
    xp = gen.xp
    _, test = yaml_utils.load_dataset(config)
    test_iter = chainer.iterators.SerialIterator(test, config.batchsize_test, repeat=False, shuffle=False)

    results_dir = args.results_dir
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

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_nir', type=str)
    parser.add_argument('--dir_rgb', type=str)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--imlist_nir', type=str)
    parser.add_argument('--imlist_rgb', type=str)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--gen_model', type=str, required=True)
    args = parser.parse_args()

    test(args)
