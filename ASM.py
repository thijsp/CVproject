import numpy as np
import matplotlib.pyplot as plt
import Landmark
import cv2

class ASM():
    def __init__(self, mean_shape, aligned_shapes, threshold = 0.99):
        aligned_shapes = np.array([shape.to_vector() for shape in aligned_shapes])
        cov = np.cov(aligned_shapes, rowvar=0)
        eigval, eigvec = np.linalg.eigh(cov)
        idx = np.argsort(-eigval)
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]

        self.eigvals = []
        self.eigvectors = []
        for i, v in enumerate(eigval):
            self.eigvals.append(v)
            self.eigvectors.append(eigvec[:, i])
            if sum(self.eigvals)/eigval.sum() > threshold:
                break
        n=[1200, 500, 1800, 1350]
        self.mean_shape = mean_shape.to_vector()


    def fit_manual(self, img, scale):
        imgh = img.shape[0]
        mean_shape = Landmark.Landmark(self.mean_shape)
        #mean_shape = mean_shape.transform_to_center([575, 353])
        points = mean_shape.landmarks
        min_x = abs(points[:, 0].min())
        min_y = abs(points[:, 1].min())
        points = [((point[0] + min_x) * scale, (point[1] + min_y) * scale) for point in points]
        pimg = np.array([(int(p[0] * imgh + 575), int(p[1] * imgh + 353)) for p in points])
        cv2.polylines(img, [pimg], True, (0, 255, 0))

        print pimg.shape
        plt.imshow(img, cmap='gray')
        plt.show()

