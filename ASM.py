import numpy as np
import matplotlib.pyplot as plt
import Landmark
import cv2
import intitmanual
import ProcustesAnalysis
from sklearn.decomposition import PCA


class ASM(object):

    def __init__(self, org_landmarks):
        self.org_landmarks = org_landmarks
        self.tooth_nb = self.org_landmarks[0].tooth_nb

    def build_model(self):
        mean, shapes = self.gpa(self.org_landmarks)
        self.mean_shape = mean
        self.aligned_shapes = shapes
        self.PCA(shapes)

    def gpa(self, landmarks):
        return ProcustesAnalysis.GPA(landmarks)

    def PCA(self, aligned_shapes, threshold = 0.99):
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
            if sum(self.eigvals) / eigval.sum() > threshold:
                break

    def fit(self, rg, automatic):
        if automatic:
            init = self.fit_automatic(rg)
        else:
            init = self.fit_manual(rg)
        self.train()


    def fit_manual(self, rg):
        scale = self.get_avg_norm()
        man = intitmanual.ManualInit(self, self.mean_shape, self.tooth_nb)
        return man.init_manual(rg, scale)

    def get_avg_norm(self):
        norms = np.array([landmark.get_norm() for landmark in self.org_landmarks])
        avg = np.mean(norms)
        return 620

    def fit_automatic(self, rg):
        pass

    def train(self):
        T = self.aligned_shapes
        x = self.mean_shape.to_vector()
        y = np.array([t.to_vector() - x for t in T])

        #nb_comp = 3 -> 99%
        pca = PCA(n_components=3, svd_solver='arpack')

        U, S, V = pca._fit(self.aligned_shapes)
        y = self.get_deviation(T, x)

        S = self.regularisation(S)

        k = np.dot(V, np.dot(U.T, y).T)
        self.c = np.dot(k, np.linalg.inv(S))

    def regularisation(self, S):
        pass

    def get_deviation(self, aligned, mean):
        T = aligned
        x = mean
        y = np.array([np.power(t.to_vector() - x, 2) for t in T])
        y = np.sum(y, axis=0)


