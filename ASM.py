import numpy as np
import matplotlib.pyplot as plt

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

        self.mean_shape = mean_shape

    def fit(self, img):
        raise NotImplementedError
        pos = self.mean_shape + np.dot(self.eigvectors, b)