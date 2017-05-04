from __future__ import division
import numpy as np

filename = 'Data/Landmarks/original/landmarks'
file_list = np.zeros((8, 14, 2, 40))
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
            file_list[j-1, i-1, 0, :] = x
            file_list[j-1, i-1, 1, :] = y

# Procrustes Analysis
landmarks        = file_list
landmarks_zeroed = landmarks - landmarks.mean(axis=0)
mean_shape       = landmarks_zeroed[0]

norm = np.power(np.linalg.norm(mean_shape),2)
PA_landmarks = []
for example in landmarks_zeroed:
    a  = np.dot(np.array(zip(mean_shape[0], mean_shape[1])).ravel(),
                np.array(zip(example[0], example[1])).ravel()) / norm
    x1 = mean_shape[0]
    x2 = example[0]
    y1 = mean_shape[1]
    y2 = example[1]
    b  = np.sum(x1*y2-y1*x2) / norm

    s = np.sqrt(a ** 2 + b ** 2)
    theta = np.tanh(b / a)

    x = s * (np.cos(theta) * x2 - np.sin(theta) * y2)
    y = s * (np.sin(theta) * x2 - np.cos(theta) * y2)

    PA_landmarks.append([x, y])



