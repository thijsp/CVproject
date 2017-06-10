import numpy as np
import matplotlib.image as mimpg
import DataReader
import Plot
import cv2
import preprocessing
import split
import matplotlib.pyplot as plt


class Radiograph(object):

    def __init__(self, source):
        if len(source) != 1:
            self.img = source
        else:
            self.img = DataReader.read_radiograph(source[0])
        #self.img = self.resize(1200, 800)
        self.img = self.to_gray()
        self.roi = preprocessing.get_roi(self.img)
        self.boundary_size = 50
        self.upper, self.lower, self.middle = self.get_jaws(self.boundary_size)

    def show(self):
        print self.img.shape
        Plot.plot_image_gray(self.img)

    def to_gray(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def get_jaws(self, boundary_size):
        img = self.roi
        #img = preprocessing.preprocess(img)
        upper, lower, middle = split.split(img, boundary_size)
        upper = preprocessing.preprocess(upper)
        lower = preprocessing.preprocess(lower)
        return upper, lower, middle

    def resize(self, width, height):
        scale = min(float(width) / self.img.shape[1], float(height) / self.img.shape[0])
        img = cv2.resize(self.img, (int(self.img.shape[1] * scale), int(self.img.shape[0] * scale)))
        #return Radiograph(img), scale
        return img

    def get_tooth_img(self, tooth_nb):
        if tooth_nb < 4:
            return self.upper
        else:
            return self.lower


if __name__ == '__main__':
    for i in range(1, 15):
        rg = Radiograph([i])
        rg.get_jaws()
