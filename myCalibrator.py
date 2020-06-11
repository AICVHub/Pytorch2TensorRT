# -*- coding: utf-8 -*-
"""
Created on : 20200608
@author: LWS

Create custom calibrator, use to calibrate int8 TensorRT model.

Need to override some methods of trt.IInt8EntropyCalibrator2, such as get_batch_size, get_batch,
read_calibration_cache, write_calibration_cache.

"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import os
import numpy as np
from PIL import Image

import torchvision.transforms as transforms

class CenterNetEntropyCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, args, files_path='/home/user/Downloads/datasets/train_val_files/val.txt'):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = 'CenterNet.cache'

        self.batch_size = args.batch_siz
        self.Channel = args.channel
        self.Height = args.height
        self.Width = args.width
        self.transform = transforms.Compose([
            transforms.Resize([self.Height, self.Width]),  # [h,w]
            transforms.ToTensor(),
        ])

        self._txt_file = open(files_path, 'r')
        self._lines = self._txt_file.readlines()
        np.random.shuffle(self._lines)
        self.imgs = [os.path.join('/home/user/Downloads/datasets/train_val_files/images',
                                  line.rstrip() + '.jpg') for line in self._lines]
        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs)//self.batch_size
        self.data_size = trt.volume([self.batch_size, self.Channel,self.Height, self.Width]) * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)

    def next_batch(self):
        if self.batch_idx < self.max_batch_idx:
            batch_files = self.imgs[self.batch_idx * self.batch_size:\
                                    (self.batch_idx + 1) * self.batch_size]
            batch_imgs = np.zeros((self.batch_size, self.Channel, self.Height, self.Width),
                                  dtype=np.float32)
            for i, f in enumerate(batch_files):
                img = Image.open(f)
                img = self.transform(img).numpy()
                assert (img.nbytes == self.data_size/self.batch_size), 'not valid img!'+f
                batch_imgs[i] = img
            self.batch_idx += 1
            print("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
            return np.ascontiguousarray(batch_imgs)
        else:
            return np.array([])

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        try:
            batch_imgs = self.next_batch()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size*self.Channel*self.Height*self.Width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32))
            return [int(self.device_input)]
        except:
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
