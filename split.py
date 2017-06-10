import numpy as np
import cv2
import radiograph
import matplotlib.pyplot as plt
import preprocessing


def upper(radiograph, l, b):
    return radiograph[:l+b, :]


def lower(radiograph, l, b):
    return radiograph[l-b:, :]


def split(radiograph, boundary_size):
    #dst = cv2.reduce(radiograph, 0, cv2.cv.CV_REDUCE_SUM, dtype=cv2.CV_32S)
    #hist = cv2.calcHist(radiograph, [0], None, [256], [0, 256])
    pre_img = preprocessing.split_processing(radiograph)
    l = middle_idx(pre_img)
    return upper(radiograph, l, boundary_size), lower(radiograph, l, boundary_size), l


def get_low_level(rg):
    low = np.min(rg)
    high = np.max(rg)
    f = (high - low) * 0.1
    return low + f


def low_intensity_index(rg):
    idx = np.where(rg < get_low_level(rg))
    return idx


def middle_idx(rg):
    idx = low_intensity_index(rg)
    y_size, x_size = rg.shape
    y_middle = int(y_size/2)
    y_idx = idx[0]
    y_idx = y_idx[(y_idx < y_middle + 100) & (y_idx > y_middle - 100)]
    #y = idx[0]
    #y = y[y_idx]
    l = np.median(y_idx)
    return int(l)

