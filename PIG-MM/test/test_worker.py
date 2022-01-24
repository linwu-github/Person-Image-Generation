
import os
import sys, os

sys.path.insert(0, os.path.join('..', '..'))
proj_root = os.path.join('..', '..')
data_root = os.path.join(proj_root, 'Data')
model_root = os.path.join(proj_root, 'Models')
save_root = os.path.join(proj_root, 'Results')
import numpy as np
import argparse, os
import torch, h5py
import torch.nn as nn
from collections import OrderedDict

from HDGan.proj_utils.local_utils import mkdirs
from HDGan.HDGan_test import test_gans


from HDGan.models.hd_networks import Generator

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Gans')

    parser.add_argument('--batch_size', type=int, default=36, metavar='N',
                        help='batch size.')
    parser.add_argument('--device_id', type=int, default=0,
                        help='which device')
    parser.add_argument('--load_from_epoch', type=int, default=500,
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, default='HDGAN_256_birds')
    parser.add_argument('--dataset', type=str, default='birds',
                        help='which dataset to use [birds or flowers]')
    parser.add_argument('--noise_dim', type=int, default=100, metavar='N',
                        help='the dimension of noise.')
    parser.add_argument('--finest_size', type=int, default=256, metavar='N',
                        help='target image size.')
    parser.add_argument('--test_sample_num', type=int, default=10,
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--save_visual_results', action='store_true',
                        help='if save visual results in folders')

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    if args.finest_size <= 256:
        netG = Generator(sent_dim=768, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, num_resblock=1)
    else:
        from HDGan.models.hd_networks import GeneratorSuperL1Loss

        netG = GeneratorSuperL1Loss(sent_dim=768, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, num_resblock=2)

    datadir = os.path.join(data_root, args.dataset)

    device_id = getattr(args, 'device_id', 0)

    if args.cuda:
        netG = netG.cuda(device_id)
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = True

    data_name = "CUHK_PEDES"
    datadir = 'D:/CODE_ALL/HDGan-master/Data/CUHK_PEDES'

    import torchvision.transforms as transforms

    image_transform = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip()])
    from dataset_CUHK_test import TextDataset

    split_dir = 'test'
    dataset = TextDataset(datadir, split=split_dir,
                          base_size_1=64,
                          base_size_2=32,
                          transform=image_transform)
    model_name = '{}_{}'.format(args.model_name, data_name)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        drop_last=True,
        shuffle=False, num_workers=4)


    test_gans(dataloader, netG, args)


