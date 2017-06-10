import numpy as np
import cv2
import sklearn
import radiograph
import Landmark
import matplotlib.pyplot as plt
import line_iterator

class GreyLevel(object):

    def __init__(self, landmarks, radiographs, nb_pixels=10):
        self.landmarks = landmarks
        self.rgs = radiographs
        self.nb_pixels = nb_pixels
        self.model_mu, self.model_cov = self._build_stat_model(self._build_model())

    def _build_model(self):
        g = []
        for i in range(len(self.landmarks)):
            landmark, jaw = self.get_example(self.landmarks[i], self.rgs[i])
            g.append(self.get_sample(landmark, jaw, self.nb_pixels)[:, :, 2])
        return np.array(g)

    def get_sample(self, landmark, jaw, nb_pixels):
        g = []
        for l in range(len(landmark.landmarks)):
            g.append(self.sample_point(landmark, jaw, l, nb_pixels))
        g = np.array(g)
        return np.array(g)

    def sample_point(self, landmark, img, l, nb_pixels):
        a, b = line_iterator.get_normal(landmark, l)
        derivatives = line_iterator.get_derivates(a, landmark.landmarks[l], img, nb_pixels)
        return derivatives

    def get_example(self, landmark, rg):
        jaw = rg.get_tooth_img(landmark.tooth_nb)
        landmark = landmark.translate_jaw(rg.middle, rg.boundary_size)
        return landmark, jaw

    def _build_stat_model(self, g):
        mu = []
        cov = []
        for i in range(40):
            p = g[:, i, :]
            mu.append(np.mean(p))
            cov.append(np.cov(p.T))
        return np.array(mu), np.array(cov)

    def fit_profile(self, estimate, rg):
        _, jaw = self.get_example(estimate, rg)
        g_s = self.get_sample(estimate, jaw, 2*self.nb_pixels)
        f = []
        for i in range(g_s.shape[0]):
            f.append(self.best_fit(g_s[i], i, estimate))
        return Landmark.Landmark(np.array(f), estimate.tooth_nb)

    def best_fit(self, points, landmark_idx, estimate):
        fit = []
        k = self.nb_pixels
        m =  (len(points)-1)/2
        ins = points[:, 2]
        for i in np.arange(k, 2*m+1-k, 1):
            fit.append(self.get_sample_fit(points[i-k:i+k+1], landmark_idx, estimate.landmarks[landmark_idx, :]))
        fit = np.array(fit)
        best = np.argmin(fit)
        best_coord = points[best, 0:2]
        return best_coord

    def get_sample_fit(self, g_s, landmark_idx, point):
        distance = self.get_distance(point, g_s)
        g_s = g_s[:, 2]
        i_2 = g_s - self.model_mu[landmark_idx]
        i_1 = i_2.T
        inv_cov = np.linalg.inv(self.model_cov[landmark_idx])
        f_gs = np.dot(np.dot(i_1, inv_cov), i_2)
        if distance > 10:
            f_gs = float('inf')

        return f_gs

    def get_distance(self, point, candidates):
        nb = candidates.shape[0] // 2 + 1
        dist = np.sqrt((candidates[nb, 0] - point[0]) ** 2 + (candidates[nb, 1] - point[1]) ** 2)
        return dist


if __name__ == '__main__':
    rg = []
    j = 6
    marks = []
    for i in np.arange(1, 15, 1):
        rg.append(radiograph.Radiograph([i]))
        landmark = (Landmark.Landmark(mark=[i, j], tooth_nb=j))
        marks.append(landmark)


    model = GreyLevel(marks, rg)