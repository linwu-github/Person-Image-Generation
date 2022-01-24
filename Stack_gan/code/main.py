from __future__ import print_function
from miscc.config import cfg, cfg_from_file

import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)



def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/cuhk_3stages.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='./Data/')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 42
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '/opt/caoshujian/stack_gan/output/%s_%s' % (cfg.DATASET_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        if cfg.DATASET_NAME == 'birds':
            bshuffle = False
            split_dir = 'test'

    # Get data loader
    imsize_1 = cfg.TREE.BASE_SIZE_1 * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    imsize_2 = cfg.TREE.BASE_SIZE_2 * (2 ** (cfg.TREE.BRANCH_NUM - 1))

    image_transform = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip()])
    from dataset_CUHK import TextDataset
    data_dir = args.data_dir
    dataset = TextDataset(data_dir, split=split_dir,
                          base_size_1=cfg.TREE.BASE_SIZE_1,
                          base_size_2=cfg.TREE.BASE_SIZE_2,
                          transform=image_transform)

    assert dataset
    num_gpu = len(cfg.GPU_ID.split(','))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
        drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    if not cfg.GAN.B_CONDITION:
        from trainer import GANTrainer as trainer
    else:
        from trainer import condGANTrainer as trainer
    algo = trainer(output_dir, dataloader, imsize_1)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.evaluate(split_dir)
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
    # Running time comparison for 10epoch with batch_size 24 on birds dataset
    #    T(1gpu) = 1.383 T(2gpus)
    #        - gpu 2: 2426.228544 -> 4min/epoch
    #        - gpu 2 & 3: 1754.12295008 -> 2.9min/epoch
    #        - gpu 3: 2514.02744293
