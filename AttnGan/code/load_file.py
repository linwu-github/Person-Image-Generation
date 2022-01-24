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

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict


def load_captions(data_dir, split):
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

def build_dictionary(train_captions, test_captions):
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

if __name__ == "__main__":
    filepath = 'D:/NEW_HDGAN/AttnGAN-master/data/CUHK-PEDES/captions.pickle'
    with open(filepath, 'rb') as f:
        x = pickle.load(f)
        a = x[1]

    print('/' *8)