import numpy as np
import cv2
import cv
import matplotlib.pyplot as plt
import Landmark
import preprocessing

class ManualInit(object):

    def __init__(self, asm, mean_shape, tooth_nb):
        self.asm = asm
        self.mean_shape = mean_shape
        self.tooth_nb = tooth_nb

    def __click_center(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print x, y
            self.center = [x, y]

    def init_manual(self, rg, scale):
        # get the correct jaw and plot
        img = self.get_img(rg)
        cv2.imshow('choose', img)
        plt.show()

        cv.SetMouseCallback('choose', self.__click_center)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # plot the result of the click
        img = self.plot_result(img, scale)
        return Landmark.Landmark(img, self.mean_shape.tooth_nb)


    def plot_result(self, img, scale):
        mean_shape = self.mean_shape.transform_unit()
        mean_shape = mean_shape.scale(scale)
        mean_shape = mean_shape.transform_to_center(self.center)
        points = mean_shape.landmarks
        pimg = np.array([(int(p[0]), int(p[1])) for p in points])

        plt.imshow(img, cmap='gray')
        cv2.polylines(img, [pimg], True, (0, 255, 0))
        plt.show()

        return pimg

    def get_img(self, rg):
        return rg.get_tooth_img(self.tooth_nb)
