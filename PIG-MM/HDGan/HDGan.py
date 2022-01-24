import numpy as np
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.nn import Parameter
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm
# from .proj_utils.plot_utils import *
from .proj_utils.torch_utils import *
import time
import json
import functools
from copy import deepcopy
from torch.optim import lr_scheduler

import torchvision.utils as vutils
def save_img_results(imgs_tcpu, fake_imgs, num_imgs, count, image_dir):
    num = 64
    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(real_img, '%s/real_samples.png' % (image_dir), normalize=True)

    for i in range(num_imgs):
        fake_img = fake_imgs[i][0:num]
        vutils.save_image(fake_img.data, '%s/count_%09d_fake_samples%d.png' % (image_dir, count, i), normalize=True)



def to_img_dict_(*inputs, super512=False):
    
    if type(inputs[0]) == tuple:
        inputs = inputs[0]
    res = {}
    res['output_64'] = inputs[0]
    res['output_128'] = inputs[1]
    res['output_256'] = inputs[2]
    # generator returns different things for 512HDGAN
    if not super512:
        # from Generator
        mean_var = (inputs[3], inputs[4])
        loss = mean_var
    else:
        # from GeneratorL1Loss of 512HDGAN
        res['output_512'] = inputs[3]
        l1loss = inputs[4] # l1 loss
        loss = l1loss

    return res, loss

def imshow(img):
    img = img / 2 + 0.5
    img = np.transpose(img.numpy(), (1, 2, 0))
    plt.imshow(img)

def get_KL_Loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    kld = mu.pow(2).add(logvar.mul(2).exp()).add(-1).mul(0.5).add(logvar.mul(-1))
    kl_loss = torch.mean(kld)
    return kl_loss

def compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels):

    criterion = nn.MSELoss()
    real_d_loss = criterion(real_logit, real_labels)
    wrong_d_loss = criterion(wrong_logit, fake_labels)
    fake_d_loss = criterion(fake_logit, fake_labels)

    discriminator_loss = real_d_loss + (wrong_d_loss+fake_d_loss) / 2.
    return discriminator_loss



def compute_d_pair_mixup_loss(mix_up_logits, mix_up_labels):

    criterion = nn.MSELoss()
    mix_up_d_loss = criterion(mix_up_logits, mix_up_labels)

    return mix_up_d_loss

def compute_d_mixup_loss(mixup_logit,  mixup_labels):
    criterion = nn.MSELoss()
    mixup_d_loss = criterion(mixup_logit, mixup_labels)
    return mixup_d_loss



def compute_d_img_loss(wrong_img_logit, real_img_logit, fake_img_logit, real_labels, fake_labels):

    criterion = nn.MSELoss()
    wrong_d_loss = criterion(wrong_img_logit, real_labels)
    real_d_loss = criterion(real_img_logit, real_labels)
    fake_d_loss = criterion(fake_img_logit, fake_labels)

    return fake_d_loss + (wrong_d_loss+real_d_loss) / 2

def compute_d_img_loss_2(wrong_img_logit, real_labels):

    criterion = nn.MSELoss()
    wrong_d_loss = criterion(wrong_img_logit, real_labels)

    return wrong_d_loss / 2

def compute_g_loss(fake_logit, real_labels):

    criterion = nn.MSELoss()
    generator_loss = criterion(fake_logit, real_labels)
    return generator_loss

def prepare_data(data):
    imgs, w_imgs, t_embedding, file_name, label = data

    real_vimgs, wrong_vimgs = [], []

    vembedding = Variable(t_embedding).cuda()

    label = Variable(label).cuda()

    for i in range(3):   # 3 = num_Dis

        real_vimgs.append(Variable(imgs[i]).cuda())
        wrong_vimgs.append(Variable(w_imgs[i]).cuda())

    return imgs, real_vimgs, wrong_vimgs, vembedding, file_name, label

def train_gans_PIG_MM(dataloader, model_root, model_name, netG, netD, id_net, teacher_net, args):
    """
    Parameters:
    ----------
    dataset: 
        data loader. refers to fuel.dataset
    model_root: 
        the folder to save the model weights
    model_name : 
        the model_name 
    netG:
        Generator
    netD:
        Descriminator
    """
    print('PIG-MM is training ')
    d_lr = args.d_lr
    g_lr = args.g_lr
    id_lr = args.id_loss_lr
    tot_epoch = args.maxepoch + 1

    ''' configure optimizer '''
    optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(0.5, 0.999))

    optim_name = optim.SGD
    ignored_params = list(map(id, id_net.module.fc.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, id_net.module.parameters())
    classifier_params = id_net.module.fc.parameters()
    optimizer_id_net = optim_name([
        {'params': base_params, 'lr': 0.1 * id_lr},
        {'params': classifier_params, 'lr': id_lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_id_net, step_size=args.epoch_decay, gamma=0.5)


    start_epoch = 1


    # create discrimnator label placeholder (not a good way)
    REAL_global_LABELS = Variable(torch.FloatTensor(args.batch_size, 1).fill_(1)).cuda()
    FAKE_global_LABELS = Variable(torch.FloatTensor(args.batch_size, 1).fill_(0)).cuda()
    REAL_local_LABELS = Variable(torch.FloatTensor(args.batch_size, 1, 5, 3).fill_(1)).cuda()
    FAKE_local_LABELS = Variable(torch.FloatTensor(args.batch_size, 1, 5, 3).fill_(0)).cuda()

    def get_labels(logit):
        # get discriminator labels for real and fake
        if logit.size(-1) == 1: 
            return REAL_global_LABELS.view_as(logit), FAKE_global_LABELS.view_as(logit)
        else:
            return REAL_local_LABELS.view_as(logit), FAKE_local_LABELS.view_as(logit)

    # to_img_dict = functools.partial(to_img_dict_, super512=args.finest_size == 512)
    nz = 100   # Z
    batch_size = dataloader.batch_size
    noise = Variable(torch.FloatTensor(batch_size, nz)).cuda()
    fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1)).cuda()

    predictions = []
    count = 0

    #--------Start training------------#
    for epoch in range(start_epoch, tot_epoch):
        start_timer = time.time()
        '''decay learning rate every epoch_decay epoches'''
        if epoch % args.epoch_decay == 0:
            d_lr = d_lr/2
            g_lr = g_lr/2

            set_lr(optimizerD, d_lr)
            set_lr(optimizerG, g_lr)

        netG.train()
        netD.train()
        num_Ds = 3
        image_dir = args.img_dir
        num_batches = len(dataloader)
        id_loss_rate = args.id_loss_rate


        id_net.train()
        teacher_net.eval()
        # correct = 0
        # total = 0
        gene_loss_epoch = 0
        dis_loss_epoch = 0

        id_criterion = nn.CrossEntropyLoss()
        criterion_teacher = nn.KLDivLoss(size_average=False)

        for step, data in enumerate(dataloader, 0):
            ''' Sample data '''

            imgs_tcpu, images, wrong_images, txt_embedding, _, label = prepare_data(data)


            ''' update D '''
            for p in netD.parameters(): p.requires_grad = True
            netD.zero_grad()
            noise.data.normal_(0, 1)

            fake_images, mean, var = netG(txt_embedding, noise)

            discriminator_loss = 0

            ''' iterate over image of different sizes.'''
            for key in range(num_Ds):
                this_img = to_device(images[key])
                this_wrong = to_device(wrong_images[key])
                this_fake = Variable(fake_images[key].data)

                real_logit, real_img_logit_local = netD(this_img, txt_embedding)
                wrong_logit, wrong_img_logit_local = netD(this_wrong, txt_embedding)
                fake_logit, fake_img_logit_local = netD(this_fake, txt_embedding)


                ''' compute disc pair loss '''
                real_labels, fake_labels = get_labels(real_logit)
                pair_loss = compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels)

                ''' compute disc image loss '''
                real_labels, fake_labels = get_labels(real_img_logit_local)
                img_loss = compute_d_img_loss(wrong_img_logit_local, real_img_logit_local, fake_img_logit_local,
                                            real_labels, fake_labels)

                discriminator_loss += (pair_loss + img_loss)

            discriminator_loss.backward()
            optimizerD.step()

            netD.zero_grad()
            d_loss_val = to_numpy(discriminator_loss).mean()
            dis_loss_epoch += d_loss_val




            ''' update G '''
            for p in netD.parameters(): p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            id_net.zero_grad()

            loss_val = 0

            kl_loss = get_KL_Loss(mean, var)
            kl_loss_val = to_numpy(kl_loss).mean()
            generator_loss = args.KL_COE * kl_loss

            id_loss_sum = 0
            mix_id_loss_sum = 0
            lbd = np.random.beta(args.alpha_rate, args.alpha_rate)

            '''Compute gen loss'''
            for key in range(3):
                # this_img = to_device(images[key])
                this_fake = fake_images[key]
                fake_pair_logit, fake_img_logit_local = netD(this_fake, txt_embedding)

                "compute id_loss"
                predict, _ = id_net(this_fake)
                id_loss = id_loss_rate * id_criterion(predict, label)
                _, predicted = torch.max(predict.data, 1)
                # correct += predicted.eq(label.data).cpu().sum()

                id_loss_sum += id_loss

                " compute id_loss use soft_label"
                index = torch.randperm(batch_size).cuda()
                mixed_img = lbd * this_fake + (1 - lbd) * this_fake[index, :]
                soft_label = teacher_net(mixed_img)
                p_teacher = F.softmax(soft_label, dim=1)
                predict_mix, _ = id_net(mixed_img)
                p_mix_stu = F.log_softmax(predict_mix, dim=1)
                mix_id_loss = args.soft_rate * id_loss_rate * criterion_teacher(p_mix_stu, p_teacher) / batch_size
                mix_id_loss_sum += mix_id_loss

                # -- compute pair loss ---
                real_labels, _ = get_labels(fake_pair_logit)
                generator_loss += compute_g_loss(fake_pair_logit, real_labels)

                # -- compute image loss ---
                real_labels, _ = get_labels(fake_img_logit_local)
                img_loss = compute_g_loss(fake_img_logit_local, real_labels)
                generator_loss += (img_loss + id_loss + mix_id_loss)
                
                if (key == 2) and count % 500 == 0:
                    print('id_loss_sum', id_loss_sum)
                    print('mix_id_loss_sum', mix_id_loss_sum)
                    print('generator_loss', generator_loss)


            generator_loss.backward()
            optimizerG.step()
            optimizer_id_net.step()
            netG.zero_grad()
            id_net.zero_grad()
            g_loss_val = to_numpy(generator_loss).mean()
            gene_loss_epoch += g_loss_val


            # --- visualize train samples----
            if (step+1) % args.verbose_per_iter == 0:
                print('[epoch %d/%d iter %d/%d]: lr = %.6f lr_id = %.6f  g_loss = %.5f d_loss= %.5f' %
                      (epoch, tot_epoch, step, num_batches, g_lr, id_lr,  g_loss_val, d_loss_val))
                sys.stdout.flush()

            count = count + 1
            # total += batch_size * 3

            if count % 500 == 0:

                fake_imgs, _, _ = netG(txt_embedding, fixed_noise)
                save_img_results(imgs_tcpu, fake_imgs, num_Ds, count, image_dir)

        id_lr_base = optimizer_id_net.param_groups[0]['lr']
        id_lr_fc = optimizer_id_net.param_groups[1]['lr']
        gene_loss_epoch_mean = gene_loss_epoch / num_batches
        dis_loss_epoch_mean = dis_loss_epoch / num_batches
        # correct_rate = 100. * correct / total
        print('[epoch %d/%d]: lr = %.6f id_lr_base = %.6f id_lr_fc = %.6f gene_loss = % .6f dis_loss=%.6f' %
              (epoch, tot_epoch, g_lr, id_lr_base, id_lr_fc, gene_loss_epoch_mean, dis_loss_epoch_mean))

        '''update lr'''
        exp_lr_scheduler.step()
        end_timer = time.time() - start_timer
        ''' save weights '''


        if epoch % args.save_freq == 0:
            netG = netG.cpu()
            netD = netD.cpu()
            id_net = id_net.cpu()
            netG_ = netG.module if 'DataParallel' in str(type(netD)) else netG
            netD_ = netD.module if 'DataParallel' in str(type(netD)) else netD
            id_net_ = id_net.module if 'DataParallel' in str(type(id_net)) else id_net
            torch.save(netG_.state_dict(), os.path.join(
                image_dir, 'G_epoch{}.pth'.format(epoch)))
            torch.save(netD_.state_dict(), os.path.join(
                image_dir, 'D_epoch{}.pth'.format(epoch)))
            torch.save(id_net_.state_dict(), os.path.join(
                image_dir, 'id_net{}.pth'.format(epoch)))
            print('save weights at {}'.format(image_dir))
            id_net = id_net.cuda()
            netD = netD.cuda()
            netG = netG.cuda()
        print('epoch {}/{} finished [time = {}s] ...'.format(epoch, tot_epoch, end_timer))


def train_gans_PIG_ID(dataloader, model_root, model_name, netG, netD, id_net, args):
    """
    Parameters:
    ----------
    dataset:
        data loader. refers to fuel.dataset
    model_root:
        the folder to save the model weights
    model_name :
        the model_name
    netG:
        Generator
    netD:
        Descriminator
    """
    print('PIG-ID is training ')
    d_lr = args.d_lr
    g_lr = args.g_lr
    id_lr = args.id_loss_lr
    tot_epoch = args.maxepoch + 1

    ''' configure optimizer '''
    optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(0.5, 0.999))

    optim_name = optim.SGD
    ignored_params = list(map(id, id_net.module.fc.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, id_net.module.parameters())
    classifier_params = id_net.module.fc.parameters()
    optimizer_id_net = optim_name([
        {'params': base_params, 'lr': 0.1 * id_lr},
        {'params': classifier_params, 'lr': id_lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_id_net, step_size=args.epoch_decay, gamma=0.5)

    start_epoch = 1

    # create discrimnator label placeholder (not a good way)
    REAL_global_LABELS = Variable(torch.FloatTensor(args.batch_size, 1).fill_(1)).cuda()
    FAKE_global_LABELS = Variable(torch.FloatTensor(args.batch_size, 1).fill_(0)).cuda()
    REAL_local_LABELS = Variable(torch.FloatTensor(args.batch_size, 1, 5, 3).fill_(1)).cuda()
    FAKE_local_LABELS = Variable(torch.FloatTensor(args.batch_size, 1, 5, 3).fill_(0)).cuda()

    def get_labels(logit):
        # get discriminator labels for real and fake
        if logit.size(-1) == 1:
            return REAL_global_LABELS.view_as(logit), FAKE_global_LABELS.view_as(logit)
        else:
            return REAL_local_LABELS.view_as(logit), FAKE_local_LABELS.view_as(logit)

    # to_img_dict = functools.partial(to_img_dict_, super512=args.finest_size == 512)
    nz = 100  # Z
    batch_size = dataloader.batch_size
    noise = Variable(torch.FloatTensor(batch_size, nz)).cuda()
    fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1)).cuda()

    predictions = []
    count = 0

    # --------Start training------------#
    for epoch in range(start_epoch, tot_epoch):
        start_timer = time.time()
        '''decay learning rate every epoch_decay epoches'''
        if epoch % args.epoch_decay == 0:
            d_lr = d_lr / 2
            g_lr = g_lr / 2

            set_lr(optimizerD, d_lr)
            set_lr(optimizerG, g_lr)

        netG.train()
        netD.train()
        num_Ds = 3
        image_dir = args.img_dir
        num_batches = len(dataloader)
        id_loss_rate = args.id_loss_rate

        id_net.train()

        # correct = 0
        # total = 0
        gene_loss_epoch = 0
        dis_loss_epoch = 0

        id_criterion = nn.CrossEntropyLoss()
        criterion_teacher = nn.KLDivLoss(size_average=False)

        for step, data in enumerate(dataloader, 0):
            ''' Sample data '''

            imgs_tcpu, images, wrong_images, txt_embedding, _, label = prepare_data(data)

            ''' update D '''
            for p in netD.parameters(): p.requires_grad = True
            netD.zero_grad()
            noise.data.normal_(0, 1)

            fake_images, mean, var = netG(txt_embedding, noise)

            discriminator_loss = 0

            ''' iterate over image of different sizes.'''
            for key in range(num_Ds):
                this_img = to_device(images[key])
                this_wrong = to_device(wrong_images[key])
                this_fake = Variable(fake_images[key].data)

                real_logit, real_img_logit_local = netD(this_img, txt_embedding)
                wrong_logit, wrong_img_logit_local = netD(this_wrong, txt_embedding)
                fake_logit, fake_img_logit_local = netD(this_fake, txt_embedding)

                ''' compute disc pair loss '''
                real_labels, fake_labels = get_labels(real_logit)
                pair_loss = compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels)

                ''' compute disc image loss '''
                real_labels, fake_labels = get_labels(real_img_logit_local)
                img_loss = compute_d_img_loss(wrong_img_logit_local, real_img_logit_local, fake_img_logit_local,
                                              real_labels, fake_labels)

                discriminator_loss += (pair_loss + img_loss)

            discriminator_loss.backward()
            optimizerD.step()

            netD.zero_grad()
            d_loss_val = to_numpy(discriminator_loss).mean()
            dis_loss_epoch += d_loss_val

            ''' update G '''
            for p in netD.parameters(): p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            id_net.zero_grad()

            loss_val = 0

            kl_loss = get_KL_Loss(mean, var)
            kl_loss_val = to_numpy(kl_loss).mean()
            generator_loss = args.KL_COE * kl_loss

            id_loss_sum = 0


            '''Compute gen loss'''
            for key in range(3):
                # this_img = to_device(images[key])
                this_fake = fake_images[key]
                fake_pair_logit, fake_img_logit_local = netD(this_fake, txt_embedding)

                "compute id_loss"
                predict, _ = id_net(this_fake)
                id_loss = id_loss_rate * id_criterion(predict, label)
                _, predicted = torch.max(predict.data, 1)
                # correct += predicted.eq(label.data).cpu().sum()

                id_loss_sum += id_loss

                # -- compute pair loss ---
                real_labels, _ = get_labels(fake_pair_logit)
                generator_loss += compute_g_loss(fake_pair_logit, real_labels)

                # -- compute image loss ---
                real_labels, _ = get_labels(fake_img_logit_local)
                img_loss = compute_g_loss(fake_img_logit_local, real_labels)
                generator_loss += (img_loss + id_loss)

                if (key == 2) and count % 500 == 0:
                    print('id_loss_sum', id_loss_sum)
                    print('generator_loss', generator_loss)

            generator_loss.backward()
            optimizerG.step()
            optimizer_id_net.step()
            netG.zero_grad()
            id_net.zero_grad()
            g_loss_val = to_numpy(generator_loss).mean()
            gene_loss_epoch += g_loss_val

            # --- visualize train samples----
            if (step + 1) % args.verbose_per_iter == 0:
                print('[epoch %d/%d iter %d/%d]: lr = %.6f lr_id = %.6f  g_loss = %.5f d_loss= %.5f' %
                      (epoch, tot_epoch, step, num_batches, g_lr, id_lr, g_loss_val, d_loss_val))
                sys.stdout.flush()

            count = count + 1
            # total += batch_size * 3

            if count % 500 == 0:
                fake_imgs, _, _ = netG(txt_embedding, fixed_noise)
                save_img_results(imgs_tcpu, fake_imgs, num_Ds, count, image_dir)

        id_lr_base = optimizer_id_net.param_groups[0]['lr']
        id_lr_fc = optimizer_id_net.param_groups[1]['lr']
        gene_loss_epoch_mean = gene_loss_epoch / num_batches
        dis_loss_epoch_mean = dis_loss_epoch / num_batches
        # correct_rate = 100. * correct / total
        print('[epoch %d/%d]: lr = %.6f id_lr_base = %.6f id_lr_fc = %.6f gene_loss = % .6f dis_loss=%.6f' %
              (epoch, tot_epoch, g_lr, id_lr_base, id_lr_fc, gene_loss_epoch_mean, dis_loss_epoch_mean))

        '''update lr'''
        exp_lr_scheduler.step()
        end_timer = time.time() - start_timer
        ''' save weights '''

        if epoch % args.save_freq == 0:
            netG = netG.cpu()
            netD = netD.cpu()
            id_net = id_net.cpu()
            netG_ = netG.module if 'DataParallel' in str(type(netD)) else netG
            netD_ = netD.module if 'DataParallel' in str(type(netD)) else netD
            id_net_ = id_net.module if 'DataParallel' in str(type(id_net)) else id_net
            torch.save(netG_.state_dict(), os.path.join(
                image_dir, 'G_epoch{}.pth'.format(epoch)))
            torch.save(netD_.state_dict(), os.path.join(
                image_dir, 'D_epoch{}.pth'.format(epoch)))
            torch.save(id_net_.state_dict(), os.path.join(
                image_dir, 'id_net{}.pth'.format(epoch)))
            print('save weights at {}'.format(image_dir))
            id_net = id_net.cuda()
            netD = netD.cuda()
            netG = netG.cuda()
        print('epoch {}/{} finished [time = {}s] ...'.format(epoch, tot_epoch, end_timer))


