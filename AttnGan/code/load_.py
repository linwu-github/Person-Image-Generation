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


if __name__ == "__main__":
    all_captions = []
    data_dir = 'D:/NEW_HDGAN/AttnGAN-master/data/CUHK-PEDES'
    split = 'test_caption'
    cap_path = '%s/text/%s.pickle' % (data_dir, split)
    with open(cap_path, "rb") as f:
        # captions = f.read().decode('utf8').split('\n')
        # captions = f.read().encode('utf8').decode('utf8').split('\n')
        caption = pickle.load(f, encoding="bytes")
        for i in range(len(caption)):
            cnt = 0
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
                cnt += 1
                if cnt == 2:
                    break
        a = all_captions
        print('--'*9)
