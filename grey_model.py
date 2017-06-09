import numpy as np
import cv2
import sklearn
import radiograph
import Landmark
import matplotlib.pyplot as plt
import line_iterator

class GreyLevel(object):

    def __init__(self, landmarks, radiographs, nb_pixels=30):
        self.landmarks = landmarks
        self.rgs = radiographs
        self.nb_pixels = nb_pixels
        self.g = self._build_model()

    def _build_model(self):
        g = []
        for i in range(len(self.landmarks)):
            landmark, jaw = self.get_example(i)
            g.append(self.get_sample(landmark, jaw))
            print 'picture done'
        return np.array(g)

    def get_sample(self, landmark, jaw):
        g = []
        for l in range(len(landmark.landmarks)):
            g.append(self.sample_point(landmark, jaw, l))
        return np.array(g)

    def sample_point(self, landmark, img, l):
        a, b = line_iterator.get_normal(landmark, l)
        derivatives = line_iterator.get_derivates(a, landmark.landmarks[l], img, self.nb_pixels)
        return derivatives

    def get_example(self, i):
        landmark = self.landmarks[i]
        rg = self.rgs[i]
        jaw = rg.get_tooth_img(landmark.tooth_nb)
        landmark = landmark.translate_jaw(rg.middle, rg.boundary_size)
        return landmark, jaw


if __name__ == '__main__':
    rg = []
    j = 2
    marks = []
    for i in np.arange(1, 15, 1):
        rg.append(radiograph.Radiograph([i]))
        landmark = (Landmark.Landmark(mark=[i, j], tooth_nb=j))
        marks.append(landmark)


    model = GreyLevel(marks, rg)








