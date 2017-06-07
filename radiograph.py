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

    def show(self):
        print self.img.shape
        Plot.plot_image_gray(self.img)

    def to_gray(self):
        return Radiograph(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY))

    def get_jaws(self):
        plt.imshow(preprocessing.get_roi(self.to_gray().img), cmap='gray')
        plt.show()
        upper, lower = split.split(preprocessing.get_roi(self.to_gray().img))
        upper = preprocessing.preprocess(upper)
        lower = preprocessing.preprocess(lower)
        return upper, lower


if __name__ == '__main__':
    for i in range(1, 15):
        rg = Radiograph([i])
        rg.get_jaws()
