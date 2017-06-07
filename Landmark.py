import numpy as np
import DataReader
import matplotlib.pyplot as plt
import cv2


class Landmark(object):

    def __init__(self, mark):
        if len(mark) == 2:
            self.landmarks = DataReader.read_landmark_incisor(mark[0], mark[1])
        elif len(mark) == 80:
            x = mark[:40]
            y = mark[40:]
            marks = np.array((x,y)).T
            self.landmarks = marks

        else:
            self.landmarks = mark

    def get_center(self):
        x_center = self.landmarks[0, :].min() + (self.landmarks[0, :].max() - self.landmarks.min())/2
        y_center = self.landmarks[1, :].min() + (self.landmarks[1, :].max() - self.landmarks.min())/2
        return x_center, y_center

    def get_mean(self):
        return np.mean(self.landmarks, axis=0)

    def transform_unit(self):
        landmarks = self.landmarks/np.linalg.norm(self.landmarks)
        return Landmark(landmarks)

    def transform_origin(self):
        center = self.get_center()
        landmarks = self.landmarks - center
        landmark = Landmark(landmarks)
        return landmark

    def transform_to_center(self, center):
        landmark = self.landmarks + center
        return Landmark(landmark)

    def to_vector(self):
        return np.hstack((self.landmarks[:, 0], self.landmarks[:, 1]))

    def scale(self, s):
        m = self.get_mean()
        centered = self.landmarks - m
        points = centered.dot(s) + m
        return Landmark(points)

    def rotate(self, theta):
        points = np.zeros(self.landmarks.shape)
        m = self.get_mean()
        rotation = np.array([[np.cos(theta), np.sin(theta)],
                           [-np.sin(theta), np.cos(theta)]])
        tmp = self.landmarks - m
        for i in range(len(self.landmarks)):
            points[i, :] = tmp[i, :].dot(rotation)
        return Landmark(points + m)



if __name__ == '__main__':
    landmark = Landmark([1, 1])
    l = landmark.to_vector()
    radiograph_path = 'Data/Radiographs/0' + str(1) + '.tif'

    img = plt.imread(radiograph_path)
    plt.figure()
    plt.imshow(img)
    plt.scatter(l[:40], l[40:])
    plt.show()

