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

print file_list





