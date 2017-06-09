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

        eigvals = []
        eigvectors = []
        for i, v in enumerate(eigval):
            eigvals.append(v)
            eigvectors.append(eigvec[:, i])
            if sum(eigvals) / eigval.sum() > threshold:
                break

        self.eigvals = np.array(eigvals)
        self.eigvectors = np.array(eigvectors)

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
        Q = self.eigvectors
        #Q = np.dot(np.diag(self.eigvals), self.eigvectors)

        #nb_comp = 3 -> 99%

        y = self.get_deviation(T, x)
        self.b = self.get_b(Q, y)
        for i in range(1, 15):
            reconstructed = self.reconstruct(self.aligned_shapes[i].to_vector(), self.b[i])
            self.plot_reconstructed(self.aligned_shapes[i].to_vector(), reconstructed)
        print('asgklj;')

    def get_deviation(self, aligned, mean):
        T = aligned
        x = mean

        #y = np.array([np.power(t.to_vector() - x, 2) for t in T])
        return np.array([t.to_vector() - x for t in T])
        #return np.sum(y, axis=0)

    def get_b(self, Q, y):
        pca = PCA(n_components=Q.shape[0], svd_solver='full')
        U, S, V = pca._fit(Q)
        return np.array([np.dot(np.dot(np.dot(V.T, np.diag(S)), U.T).T, diff) for diff in y])

    def reconstruct(self, shape, b):
        return self.mean_shape.to_vector() + np.dot(self.eigvectors.T, b)

    def plot_reconstructed(self, tooth, reconstructed):
        plt.scatter(tooth[:40], tooth[40:])
        plt.scatter(reconstructed[:40], reconstructed[40:])
        plt.show()