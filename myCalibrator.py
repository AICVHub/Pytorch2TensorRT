# -*- coding: utf-8 -*-
"""
Created on : 20200608
@author: LWS

Create custom calibrator, use to calibrate int8 TensorRT model.

Need to override some methods of trt.IInt8EntropyCalibrator2, such as get_batch_size, get_batch,
read_calibration_cache, write_calibration_cache.

"""
import glob

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

import os
import numpy as np


class MyEntropyCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, tensor_shape, imgs_dir,
                 mean=(0., 0., 0.), std=(255., 255., 255.)):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = 'CALIBRATOR.cache'
        self.mean = mean
        self.std = std

        self.batch_size, self.Channel, self.Height, self.Width = tensor_shape

        self.imgs = glob.glob(imgs_dir)
        np.random.shuffle(self.imgs)

        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs) // self.batch_size
        self.data_size = trt.volume([self.batch_size, self.Channel, self.Height, self.Width]) * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)

    def next_batch(self):
        if self.batch_idx < self.max_batch_idx:
            batch_files = self.imgs[self.batch_idx * self.batch_size: \
                                    (self.batch_idx + 1) * self.batch_size]
            batch_imgs = np.zeros((self.batch_size, self.Channel, self.Height, self.Width),
                                  dtype=np.float32)
            for i, f in enumerate(batch_files):
                img = cv2.imread(f)
                img = self.transform(img, (self.Width, self.Height))
                assert (img.nbytes == self.data_size / self.batch_size), 'not valid img!' + f
                batch_imgs[i] = img
            self.batch_idx += 1
            print("\rbatch:[{}/{}]".format(self.batch_idx, self.max_batch_idx), end='')
            return np.ascontiguousarray(batch_imgs)
        else:
            return np.array([])

    def transform(self, img, img_size):
        w, h = img_size
        oh, ow = img.shape[0:2]
        s = min(w / ow, h / oh)
        nw, nh = int(round(ow * s)), int(round(oh * s))
        t, b, l, r = (h - nh) // 2, (h - nh + 1) // 2, (w - nw) // 2, (w - nw + 1) // 2
        if nw != ow or nh != oh:
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
        img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=114)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - np.float32(self.mean)) * (1 / np.float32(self.std))
        img = img.transpose(2, 0, 1).copy()
        return img

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        try:
            batch_imgs = self.next_batch()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size * self.Channel * self.Height * self.Width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32))

            return [int(self.device_input)]
        except Exception as e:
            print("get batch error: {}".format(e))
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


if __name__ == '__main__':
    calib = MyEntropyCalibrator(1, 3, 384, 640)
    batch = calib.get_batch(None)
    print(batch)
