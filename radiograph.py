import numpy as np
import matplotlib.image as mimpg
import DataReader
import Plot
import cv2
import preprocessing

class Radiograph(object):

    def __init__(self, source):
        if len(source) != 1:
            self.img = source
        else:
            self.img = DataReader.read_radiograph(source)

    def show(self):
        Plot.plot_image_gray(self.img)

    def to_gray(self):
        return Radiograph(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY))

