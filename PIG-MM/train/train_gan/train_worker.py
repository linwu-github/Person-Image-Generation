# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.join('..', '..'))

proj_root = os.path.join('..', '..')
data_root = os.path.join(proj_root, 'Data')
model_root = os.path.join(proj_root, 'Models')
import torch.nn as nn
from HDGan.models.hd_networks import Generator
from HDGan.models.hd_networks import Discriminator
from HDGan.HDGan import train_gans_PIG_ID, train_gans_PIG_MM
from HDGan.resnet50_pretrain_2 import ft_net
from HDGan.ft_net_res import load_teacher_Net
import torch.utils.model_zoo as model_zoo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gans')

    parser.add_argument('--reuse_weights',   action='store_true',
                        default=False, help='continue from last checkout point')
    parser.add_argument('--load_from_epoch', type=int,
                        default=0,  help='load from epoch')

    parser.add_argument('--batch_size', type=int,
                        default=4, metavar='N', help='batch size.')
    parser.add_argument('--device_id',  type=int,
                        default=0,  help='which device')

    parser.add_argument('--model_name', type=str, default='HDGAN_256')

    parser.add_argument('--num_resblock', type=int, default=1,
                        help='number of resblock in generator')
    parser.add_argument('--epoch_decay', type=float, default=100,
                        help='decay learning rate by half every epoch_decay')
    parser.add_argument('--finest_size', type=int, default=256,
                        metavar='N', help='target image size.')
    parser.add_argument('--init_256generator_from', type=str,  default='')
    parser.add_argument('--maxepoch', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--g_lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--d_lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--id_loss_lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                        help='how frequent to save the model')
    parser.add_argument('--display_freq', type=int, default=500, metavar='N',
                        help='plot the results every {} batches')
    parser.add_argument('--verbose_per_iter', type=int, default=200,
                        help='print losses per iteration')
    parser.add_argument('--num_emb', type=int, default=4, metavar='N',
                        help='number of emb chosen for each image during training.')
    parser.add_argument('--noise_dim', type=int, default=100, metavar='N',
                        help='the dimension of noise.')
    parser.add_argument('--ncritic', type=int, default=1, metavar='N',
                        help='the channel of each image.')
    parser.add_argument('--test_sample_num', type=int, default=4,
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--KL_COE', type=float, default=4, metavar='N',
                        help='kl divergency coefficient.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='which gpu')
    parser.add_argument('--img_dir', type=str, default='D:/NEW_HDGAN/FINAL_TEST/CODE/PIG-MM/output',
                        help='image_direction')
    parser.add_argument('--data_path', type=str, default='D:\CODE_ALL\StackGAN-v2-master\data\CUHK_PEDES',
                        help='data path')
    parser.add_argument('--id_loss_rate', type=int, default=0.5,
                        help='id_loss_rate')
    parser.add_argument('--alpha_rate', type=int, default=0.2,
                        help='alpha_rate')
    parser.add_argument('--soft_rate', type=int, default=0.5,
                        help='soft_rate')
    parser.add_argument("--PIG_ID", action='store_true',
                        default=True, help ='false is PIG-MM')
    parser.add_argument("--local_rank", type=int, default=0)
    # add more
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print(args)
    
    # Generator
    if args.finest_size <= 256:
        netG = Generator(sent_dim=768, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, num_resblock=1)
    else:
        assert args.init_256generator_from != '', '256 generator need to be intialized'
        from HDGan.models.hd_networks import GeneratorSuperL1Loss
        netG = GeneratorSuperL1Loss(sent_dim=768, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, num_resblock=2, G256_weightspath=args.init_256generator_from)
    # Discriminator
    netD = Discriminator(num_chan=3, hid_dim=128, sent_dim=768, emb_dim=128)
    print('netG', netG)
    print('netD', netD)


    # we use the first 10k train sets due to limited computing resources, we also do the experiment on the whole train set
    id_net = ft_net(3206)   # 3206 is  class_num of 10k dataset  &&  11003 is class_num of whole train set
    teacher_net = load_teacher_Net(3206)


    gpus = [a for a in range(len(args.gpus.split(',')))]
    torch.cuda.set_device(gpus[0])
    args.batch_size = args.batch_size * len(gpus)
    if args.cuda:
        print('>> Parallel models in {} GPUS'.format(gpus))
        netD = nn.parallel.DataParallel(netD, device_ids=range(len(gpus)))
        netG = nn.parallel.DataParallel(netG, device_ids=range(len(gpus)))
        id_net = nn.parallel.DataParallel(id_net, device_ids=range(len(gpus)))
        if not args.PIG_ID:
            teacher_net = nn.parallel.DataParallel(teacher_net, device_ids=range(len(gpus)))
            teacher_net = teacher_net.cuda()
        netD = netD.cuda()
        netG = netG.cuda()
        id_net = id_net.cuda()


        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    data_name = "CUHK_PEDES"
    datadir = args.data_path

    import torchvision.transforms as transforms
    image_transform = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip()])
    from dataset_CUHK import TextDataset

    split_dir = 'train'
    dataset = TextDataset(datadir, split=split_dir,
                          base_size_1=64,
                          base_size_2=32,
                          transform=image_transform)
    model_name = '{}_{}'.format(args.model_name, data_name)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        drop_last=True,
        shuffle=True, num_workers=4)

    print('>> Start training ...')
    if args.PIG_ID:
        train_gans_PIG_ID(dataloader, model_root, model_name, netG, netD, id_net, args)
    else:
        train_gans_PIG_MM(dataloader, model_root, model_name, netG, netD, id_net, teacher_net, args)
