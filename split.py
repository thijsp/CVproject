import numpy as np
import cv2
import radiograph
import matplotlib.pyplot as plt


def upper(radiograph, l):
    return radiograph[:l, :]


def lower(radiograph, l):
    return radiograph[l:, :]


def split(radiograph):
    #dst = cv2.reduce(radiograph, 0, cv2.cv.CV_REDUCE_SUM, dtype=cv2.CV_32S)
    #hist = cv2.calcHist(radiograph, [0], None, [256], [0, 256])
    l = middle_idx(radiograph)
    return upper(radiograph, l), lower(radiograph, l)


def low_intensity_index(rg):
    idx = np.where(rg < 1)
    return idx

def middle_idx(rg):
    idx = low_intensity_index(rg)
    y_size, x_size = rg.shape
    y_middle = int(y_size/2)
    y_idx = idx[1]
    y_idx = np.where(y_idx [(y_idx < y_middle + 100) & (y_idx > y_middle - 100)])
    y = idx[1]
    y = y[y_idx]
    l = np.median(y)
    return int(l)