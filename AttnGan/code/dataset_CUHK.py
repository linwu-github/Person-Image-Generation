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
from miscc.config import cfg

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

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
        self.embeddings_num = 2

        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, self.wordtoix, \
                        self.n_words = self.load_text_data(data_dir, split)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))

        self.number_example = len(self.filenames)

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        if split == 'train':
            file_name = 'train_files_1w'
        else:
            file_name = 'test_files'
        filepath = '%s/%s/%s.pickle' % (data_dir, split, file_name)
        # filepath = os.path.join(data_dir, 'test_files.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')

        if not os.path.isfile(filepath):
            # 把caption 分成 word
            train_caption = self.load_captions(data_dir, 'train_caption_1W')
            test_caption = self.load_captions(data_dir, 'test_caption')

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_caption, test_caption)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath, len(filepath))
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names

        else:  # split=='test'
            captions = test_captions
            filenames = test_names

        return filenames, captions, ixtoword, wordtoix, n_words

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len


    def load_captions(self, data_dir, split):
        all_captions = []
        cap_path = '%s/text/%s.pickle' % (data_dir, split)
        with open(cap_path, "rb") as f:
            # captions = f.read().decode('utf8').split('\n')
            # captions = f.read().encode('utf8').decode('utf8').split('\n')
            caption = pickle.load(f, encoding="bytes")
            for i in range(len(caption)):
                for cap in caption[i]:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)  # 我们只需要把caption都导入就行了
        captions = train_captions + test_captions    # 统计全部的caption 中的word次数
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)
        # 这里的train_caption_new就是我们需要用到的caption， list
        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        # print('index', index)
        data_dir = self.data_dir
        #
        img_name = '%s/imgs/%s' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize_1, self.imsize_2,  self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num - 1)
        new_sent_ix = index * self.embeddings_num + sent_ix
        # print('new_sent_ix', new_sent_ix)
        caps, cap_len = self.get_caption(new_sent_ix)
        return imgs, caps, cap_len, cls_id, key


    def __len__(self):
        return len(self.filenames)

if __name__ == "__main__":
    data_dir = 'D:/NEW_HDGAN/AttnGAN-master/data/CUHK-PEDES'
    split_dir = 'test'
    imsize_1 = cfg.TREE.BASE_SIZE_1 * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    imsize_2 = cfg.TREE.BASE_SIZE_2 * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(data_dir, split=split_dir,
                          base_size_1=cfg.TREE.BASE_SIZE_1,
                          base_size_2=cfg.TREE.BASE_SIZE_2,
                          transform=image_transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, drop_last= True,
        shuffle=True, num_workers=0)
    s = dataset.__getitem__(0)
    for step, data in enumerate(dataloader, 0):
        a = data
        print('#' * 9)
    # data_iter = iter(dataloader)
    # num_batches = len(dataloader)
    # step = 0
    # while step < num_batches:
    #     # reset requires_grad to be trainable for all Ds
    #     # self.set_requires_grad_value(netsD, True)
    #
    #     ######################################################
    #     # (1) Prepare training data and Compute text embeddings
    #     ######################################################
    #     data = data_iter.next()
    #
    #     print('--'*9)
