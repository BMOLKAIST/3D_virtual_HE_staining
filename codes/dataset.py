import os
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from util import *

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        #self.task = task
        #self.opts = opts

        # Updated at Apr 5 2020
        self.to_tensor = ToTensor()

        lst_data = os.listdir(self.data_dir)
        #lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]
        #
        lst_label.sort()
        lst_input.sort()
        #
        self.lst_label = lst_label
        self.lst_input = lst_input

        #lst_data.sort()
        #self.lst_data = lst_data


    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        # label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        # input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = plt.imread(os.path.join(self.data_dir, self.lst_label[index]))
        # sz = label.shape

        input = plt.imread(os.path.join(self.data_dir, self.lst_input[index]))

        # if sz[0] > sz[1]:
        #     img = img.transpose((1, 0, 2))

        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        if label.dtype == np.uint8:
            label = label / 255.0

        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        if input.dtype == np.uint8:
            input = input / 255.0

        #if self.opts[0] == 'direction':
        #    if self.opts[1] == 0: # label: left | input: right
        #        data = {'label': img[:, :sz[1]//2, :], 'input': img[:, sz[1]//2:, :]}
        #    elif self.opts[1] == 1: # label: right | input: left
        #        data = {'label': img[:, sz[1]//2:, :], 'input': img[:, :sz[1]//2, :]}
        #else:
        #    data = {'label': img}


        #if self.task == "inpainting":
        #    data['input'] = add_sampling(data['label'], type=self.opts[0], opts=self.opts[1])
        #elif self.task == "denoising":
        #    data['input'] = add_noise(data['label'], type=self.opts[0], opts=self.opts[1])

        #if self.transform:
        #    data = self.transform(data)

        #if self.task == "super_resolution":
        #    data['input'] = add_blur(data['label'], type=self.opts[0], opts=self.opts[1])

        
        data = {'input': input, 'label': label}

        data = self.to_tensor(data)

        return data
    
class Dataset_test(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        #self.task = task
        #self.opts = opts

        # Updated at Apr 5 2020
        self.to_tensor = ToTensor()

        lst_data = os.listdir(self.data_dir)
        #lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]

        # lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]
        #
        # lst_label.sort()
        lst_input.sort()
        #
        # self.lst_label = lst_label
        self.lst_input = lst_input

        #lst_data.sort()
        #self.lst_data = lst_data


    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):
        # label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        # input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # label = plt.imread(os.path.join(self.data_dir, self.lst_label[index]))
        # sz = label.shape

        input = plt.imread(os.path.join(self.data_dir, self.lst_input[index]))

        # if sz[0] > sz[1]:
        #     img = img.transpose((1, 0, 2))

        # if label.ndim == 2:
        #     label = label[:, :, np.newaxis]

        # if label.dtype == np.uint8:
        #     label = label / 255.0

        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        if input.dtype == np.uint8:
            input = input / 255.0

        #if self.opts[0] == 'direction':
        #    if self.opts[1] == 0: # label: left | input: right
        #        data = {'label': img[:, :sz[1]//2, :], 'input': img[:, sz[1]//2:, :]}
        #    elif self.opts[1] == 1: # label: right | input: left
        #        data = {'label': img[:, sz[1]//2:, :], 'input': img[:, :sz[1]//2, :]}
        #else:
        #    data = {'label': img}


        #if self.task == "inpainting":
        #    data['input'] = add_sampling(data['label'], type=self.opts[0], opts=self.opts[1])
        #elif self.task == "denoising":
        #    data['input'] = add_noise(data['label'], type=self.opts[0], opts=self.opts[1])

        #if self.transform:
        #    data = self.transform(data)

        #if self.task == "super_resolution":
        #    data['input'] = add_blur(data['label'], type=self.opts[0], opts=self.opts[1])

        
        data = {'input': input}

        data = self.to_tensor(data)

        return data

## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        # label, input = data['label'], data['input']
        #
        # label = label.transpose((2, 0, 1)).astype(np.float32)
        # input = input.transpose((2, 0, 1)).astype(np.float32)
        #
        # data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        # Updated at Apr 5 2020
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # label, input = data['label'], data['input']
        #
        #input = (input - self.mean) / self.std
        #label = (label - self.mean) / self.std
        #
        #data = {'label': label, 'input': input}

        # Updated at Apr 5 2020
        #for key, value in data.items():
        #    data[key] = (value - self.mean) / self.std

        return data


class RandomFlip(object):
    def __call__(self, data):
        # label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            # label = np.fliplr(label)
            # input = np.fliplr(input)

            # Updated at Apr 5 2020
            for key, value in data.items():
                data[key] = np.flip(value, axis=0)

        if np.random.rand() > 0.5:
            # label = np.flipud(label)
            # input = np.flipud(input)

            # Updated at Apr 5 2020
            for key, value in data.items():
                data[key] = np.flip(value, axis=1)

        # data = {'label': label, 'input': input}

        return data


class RandomCrop(object):
  def __init__(self, shape):
      self.shape = shape

  def __call__(self, data):
    # input, label = data['input'], data['label']
    # h, w = input.shape[:2]

    h, w = data['label'].shape[:2]
    new_h, new_w = self.shape

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
    id_x = np.arange(left, left + new_w, 1)

    # input = input[id_y, id_x]
    # label = label[id_y, id_x]
    # data = {'label': label, 'input': input}

    # Updated at Apr 5 2020
    for key, value in data.items():
        data[key] = value[id_y, id_x]

    return data


class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        for key, value in data.items():
            data[key] = resize(value, output_shape=(self.shape[0], self.shape[1],
                                                    self.shape[2]))

        return data