from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mimpg
import seaborn as sns
plt.style.use(['seaborn-white','seaborn-paper'])
sns.set(font='serif')

"""
Lecture: Active Shape Models
https://www.youtube.com/watch?v=53kx_czs7Es
"""

def transform(tooth_data):
    # Implementation of Protocol 4, Appendix A of An Introduction to Active Shape Models, Coots
    mean_shape  = tooth_data[0]
    norm        = np.sqrt(np.power(np.linalg.norm(mean_shape), 2))
    mean_shape /= norm
    dev         = 10
    while dev > 0.00001:
        for i,shape in enumerate(tooth_data):
            tooth_data[i] = align_shape(mean_shape, shape)
        old_mean_shape = mean_shape
        mean_shape     = tooth_data.mean(axis=0)
        mean_shape     = align_shape(old_mean_shape, mean_shape)
        norm           = np.sqrt(np.power(np.linalg.norm(mean_shape), 2))
        mean_shape    /= norm
        dev            = np.abs(np.sum(mean_shape - old_mean_shape))

    return tooth_data

def align_shape(mean_shape, shape):
    s, theta, x1, y1 = get_parameters(mean_shape, shape)
    # Apply scaling and rotation
    x                = s * (np.cos(theta) * x1 - np.sin(theta) * y1)
    y                = s * (np.sin(theta) * x1 + np.cos(theta) * y1)
    return [x, y]

def get_parameters(mean_shape, example):
    norm = np.power(np.linalg.norm(example), 2)
    a    = np.dot(mean_shape.T.ravel(), example.T.ravel()) / norm
    x1   = example[0]
    x2   = mean_shape[0]
    y1   = example[1]
    y2   = mean_shape[1]
    b    = np.sum(x1 * y2 - y1 * x2) / norm

    s     = np.sqrt(a ** 2 + b ** 2)
    theta = np.tanh(b / a)

    return s, theta, x1, y1

def plot_radiograph(radiograph_path, landmarks):
    img = mimpg.imread(radiograph_path)
    x_all = np.array([])
    y_all = np.array([])
    for i in range(landmarks.shape[0]):
        x, y = zip(*zip(*landmarks[i]))
        x_all = np.append(x_all, list(x))
        y_all = np.append(y_all, list(y))

    img_plot = plt.imshow(img)
    plt.scatter(x_all, y_all)
    plt.show()

def plot_radiographs():
    for i in range(1, 15):
        if i < 10:
            radiograph_path = 'Data/Radiographs/0' + str(i) + '.tif'
        else:
            radiograph_path = 'Data/Radiographs/' + str(i) + '.tif'
        plot_radiograph(radiograph_path, landmarks[:, i-1, :, :])

if __name__ == '__main__':
    # Load Data
    filename  = 'Data/Landmarks/original/landmarks'
    landmarks = np.zeros((8, 14, 2, 40))
    for i in range(1, 15):
        for j in range(1, 9):
            curr_fil = filename + str(i) + '-' + str(j) + '.txt'
            x = []
            y = []
            with open(curr_fil) as f:
                count = 0
                for line in f:
                    if np.mod(count, 2) == 0:
                        x.append(np.int32(np.float32(line)))
                    else:
                        y.append(np.int32(np.float32(line)))
                    count += 1
                landmarks[j - 1, i - 1, 0, :] = x
                landmarks[j - 1, i - 1, 1, :] = y

    # plot the radiograph with landmarks on it
    # plot_radiographs()

    # Procrustes Analysis
    shape             = landmarks.shape
    landmarks_zeroed  = landmarks - landmarks.mean(axis=1).reshape(shape[0], 1, shape[2], shape[3])
    landmarks_aligned = np.zeros(landmarks_zeroed.shape)
    for i,tooth_data in enumerate(landmarks_zeroed):
        landmarks_aligned[i] = transform(tooth_data)

    # PCA
    for i,tooth_data in enumerate(landmarks_aligned):
        tooth_data  = tooth_data.reshape(14, 80)
        eigs, eigvs = cv2.PCACompute(tooth_data)
        plt.bar(np.arange(eigs.shape[1]), eigs[0])
        plt.xlabel("Principal Components")
        plt.ylabel("Size")
        plt.title("PCA Results")


