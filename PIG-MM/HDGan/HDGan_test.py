
from .proj_utils.local_utils import *
from .proj_utils.torch_utils import *

from PIL import Image


def prepare_data(data):
    imgs, t_embedding, file_name, label = data

    vembedding = Variable(t_embedding).cuda()

    label = Variable(label).cuda()
    image = Variable(imgs[2]).cuda()

    return image, vembedding, file_name, label


def save_singleimages(images, filenames, save_dir, split_dir, feature_num, imsize):
    for i in range(images.size(0)):
        a = filenames[i].split('/')[1]
        image_name = a.split('.')[0]
        s_tmp = '%s/new_sampels/select_img/%s/%s' % (save_dir, split_dir, image_name)
        folder = s_tmp[:s_tmp.rfind('/')]
        if not os.path.isdir(folder):
            print('Make a new folder: ', folder)
            os.makedirs(folder)

        fullpath = '%s_%d_%s.png' % (s_tmp, feature_num, imsize)
        img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
        ndarr = img.permute(1, 2, 0).data.cpu().numpy()
        im = Image.fromarray(ndarr)
        im.save(fullpath)


def test_gans(dataloader, netG, args):
    # helper function
    netG.eval()
    ''' load model '''
    G_weightspath = './PIG_MM_model.pth'

    print('reload weights from {}'.format(G_weightspath))
    weights_dict = torch.load(G_weightspath, map_location=lambda storage, loc: storage)
    # load_partial_state_dict(netG, weights_dict)
    sample_weight_name = [a for a in weights_dict.keys()][0]
    if 'module' in sample_weight_name:  # if the saved model is wrapped by DataParallel.
        netG = nn.parallel.DataParallel(netG, device_ids=[0])
    ## TODO note that strict is set to false for now. It is a bit risky
    netG.load_state_dict(weights_dict, strict=False)
    netG.eval()

    testing_z = torch.FloatTensor(args.batch_size, args.noise_dim)
    testing_z = to_device(testing_z)


    count = 0
    split_dir = '3W_PIG-ID_G500_01_001'   # any name you want
    save_dir = './output'

    import time
    since = time.time()
    for i in range(4):
        print(i)
        for step, data in enumerate(dataloader, 0):
            with torch.no_grad():
                # testing_z.normal_(0, 1)
                real_img, txt_embedding, _, filenames, label = prepare_data(data)
                testing_z.normal_(0, 1)
                fake_images, _, _ = netG(txt_embedding, testing_z)
                save_singleimages(fake_images[2], filenames, save_dir, split_dir, i, 256)
                count += 1


    print('count', count)
    now = time.time()
    endtime = now - since
    print('time', endtime)

