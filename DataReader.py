from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mimpg


def read_landmarks():
    filename = 'Data/Landmarks/original/landmarks'
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
    return landmarks


def read_landmark(landmark_number):
    filename = 'Data/Landmarks/original/landmarks' + str(landmark_number)
    landmarks = np.zeros((8, 2, 40))
    for j in range(1, 9):
        curr_fil = filename + '-' + str(j) + '.txt'
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
            landmarks[j - 1, 0, :] = x
            landmarks[j - 1, 1, :] = y
    return landmarks


def read_landmark_incisor(landmark_number, incisor_number):
    filename = 'Data/Landmarks/original/landmarks' + str(landmark_number) + '-' + str(incisor_number) +'.txt'
    landmarks = np.zeros((2, 40))
    x = []
    y = []
    with open(filename) as f:
        count = 0
        for line in f:
            if np.mod(count, 2) == 0:
                x.append(np.int32(np.float32(line)))
            else:
                y.append(np.int32(np.float32(line)))
            count += 1
        landmarks[0, :] = x
        landmarks[1, :] = y
    return landmarks.T


def read_radiograph(rad_number):
    radiograph_path = 'Data/Radiographs/'
    if rad_number < 10:
        radiograph_path = radiograph_path + '0' + str(rad_number) + '.tif'
    else:
        radiograph_path = radiograph_path + str(rad_number) + '.tif'
    img = mimpg.imread(radiograph_path)
    return img



