import numpy as np
import cv2
import cv
import matplotlib.pyplot as plt
import Landmark

class ManualInit(object):

    def __init__(self, asm, mean_shape):
        self.asm = asm
        self.mean_shape = mean_shape

    def __click_center(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print x, y
            self.center = [x, y]

    def init_manual(self, rg):
        cv2.imshow('choose', rg)
        # cv2.resizeWindow('image', 60,60)
        plt.show()

        cv.SetMouseCallback('choose', self.__click_center)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        self.plot_result(rg)


    def plot_result(self, img):
        imgh = img.shape[1]
        # mean_shape = mean_shape.transform_to_center([575, 353])
        points = self.mean_shape.landmarks
        min_x = abs(points[:, 0].min())
        min_y = abs(points[:, 1].min())
        # points = [((point[0] + min_x) * scale, (point[1] + min_y) * scale) for point in points]
        pimg = np.array([(int(p[0] * imgh + self.center[1]), int(p[1] * imgh + self.center[0])) for p in points])
        cv2.polylines(img, [pimg], True, (0, 255, 0))

        print pimg.shape
        plt.imshow(img, cmap='gray')
        plt.show()
