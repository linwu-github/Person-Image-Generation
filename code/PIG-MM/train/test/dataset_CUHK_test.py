from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torchvision.transforms as transforms
import torch.utils.data as data
import os
import random
import numpy as np
import pandas as pd
import six
import sys
import torch
from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle




IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imgs(img_path, imsize_1, imsize_2, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(3):
        if i < (2):
            re_img = transforms.Resize((imsize_1[i], imsize_2[i]))(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, base_size_1=64, base_size_2=32, split='train', transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.target_transform = target_transform   # 给embedding做transform

        self.imsize_1 = []
        self.imsize_2 = []
        for i in range(3):
            self.imsize_1.append(base_size_1)
            base_size_1 = base_size_1 * 2
            self.imsize_2.append(base_size_2)
            base_size_2 = base_size_2 * 2

        self.data = []
        self.data_dir = data_dir

        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.id_embedding = self.load_id_embedding(split_dir)

        self.iterator = self.prepair_test_pairs



    def load_embedding(self, data_dir):

        embedding_filename = '/test_embedding.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f, encoding="bytes")
            a = embeddings[1]
            # embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            # print('embeddings: ', embeddings[1].shape)
        return embeddings
    def load_id_embedding(self, data_dir):

        id_embedding_filename = '/test_id_embedding.pickle'

        with open(data_dir + id_embedding_filename, 'rb') as f:
            embeddings = pickle.load(f, encoding="bytes")

            # embedding_shape = [embeddings.shape[-1]]
            print('id_embeddings: ', embeddings.shape)
        return embeddings


    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/test_labels.pickle'):
            with open(data_dir + '/test_labels.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'test_files.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        data_dir = self.data_dir
        idx = np.random.randint(0, 2)
        label = self.class_id[index] - 1
        embeddings = self.embeddings[index][idx]
        img_name = '%s/CUHK-PEDES/imgs/%s' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize_1, self.imsize_2, self.transform, normalize=self.norm)

        if self.target_transform is not None:
            embeddings = self.target_transform(embeddings)

        return imgs, embeddings.squeeze(0), key, label

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)

