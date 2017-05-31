from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mimpg
import seaborn as sns
plt.style.use(['seaborn-white','seaborn-paper'])
sns.set(font='serif')

def procrustes_analysis(landmarks):
    shape             = landmarks.shape
    landmarks_zeroed  = landmarks - landmarks.mean(axis=1).reshape(shape[0], 1, shape[2], shape[3])
    landmarks_aligned = np.zeros(landmarks_zeroed.shape)
    for i,tooth_data in enumerate(landmarks_zeroed):
        landmarks_aligned[i] = transform(tooth_data)
    return landmarks_aligned

def transform(tooth_data):
    # Implementation of Protocol 4, Appendix A of An Introduction to Active Shape Models, Coots
    mean_shape = tooth_data[0]
    norm = np.sqrt(np.power(np.linalg.norm(mean_shape), 2))
    mean_shape /= norm
    dev = 10
    while dev > 0.00001:
        for i, shape in enumerate(tooth_data):
            tooth_data[i] = align_shape(mean_shape, shape)
        old_mean_shape = mean_shape
        mean_shape = tooth_data.mean(axis=0)
        mean_shape = align_shape(old_mean_shape, mean_shape)
        norm = np.sqrt(np.power(np.linalg.norm(mean_shape), 2))
        mean_shape /= norm
        dev = np.abs(np.sum(mean_shape - old_mean_shape))
    return tooth_data


def align_shape(mean_shape, shape):
    s, theta, x1, y1 = get_parameters(mean_shape, shape)
    # Apply scaling and rotation
    x = s * (np.cos(theta) * x1 - np.sin(theta) * y1)
    y = s * (np.sin(theta) * x1 + np.cos(theta) * y1)
    return [x, y]


def get_parameters(mean_shape, example):
    norm = np.power(np.linalg.norm(example), 2)
    a = np.dot(mean_shape.T.ravel(), example.T.ravel()) / norm
    x1 = example[0]
    x2 = mean_shape[0]
    y1 = example[1]
    y2 = mean_shape[1]
    b = np.sum(x1 * y2 - y1 * x2) / norm

    s = np.sqrt(a ** 2 + b ** 2)
    theta = np.tanh(b / a)

    return s, theta, x1, y1


def perform_pca(landmarks_aligned):
    for i, tooth_data in enumerate(landmarks_aligned):
        tooth_data = tooth_data.reshape(14, 80)
        eigs, eigvs = cv2.PCACompute(tooth_data)
        plt.bar(np.arange(eigs.shape[1]), eigs[0])
        plt.xlabel("Principal Components")
        plt.ylabel("Size")
        plt.title("PCA Results")
    return eigs, eigvs


