import numpy as np
import DataReader
import matplotlib.pyplot as plt
import cv2
import radiograph


class Landmark(object):

    def __init__(self, mark, tooth_nb):
        if len(mark) == 2:
            self.landmarks = DataReader.read_landmark_incisor(mark[0], mark[1])
        elif len(mark) == 80:
            x = mark[:40]
            y = mark[40:]
            marks = np.array((x,y)).T
            self.landmarks = marks

        else:
            self.landmarks = mark
        self.tooth_nb = tooth_nb

    def get_center(self):
        #x_center = self.landmarks[:, 0].min() + (self.landmarks[:, 0].max() - self.landmarks[:, 0].min())/2
        #y_center = self.landmarks[:, 1].min() + (self.landmarks[:, 1].max() - self.landmarks[:, 1].min())/2
        x_center, y_center = np.mean(self.landmarks, axis=0)
        return np.array([x_center, y_center])

    def get_mean(self):
        return np.mean(self.landmarks, axis=0)

    def transform_unit(self):
        landmarks = self.landmarks/np.linalg.norm(self.landmarks)
        return Landmark(landmarks, self.tooth_nb)

    def transform_origin(self):

        #center = self.get_center()
        center = np.mean(self.landmarks, axis=0)
        landmarks = self.landmarks - center
        landmark = Landmark(landmarks, self.tooth_nb)
        return landmark

    def transform_to_point(self, point):
        landmarks = self.landmarks + point
        landmark = Landmark(landmarks, self.tooth_nb)
        return landmark

    def transform_to_center(self, center):
        #landmark = self.landmarks + center
        #dist = self.distance_to(self.get_center())
        new = self.transform_origin()
        landmark = new.landmarks + center
        #landmark = center + dist
        L = Landmark(landmark, self.tooth_nb)
        return L

    def distance_to(self, point):
        return self.landmarks - point

    def to_vector(self):
        return np.hstack((self.landmarks[:, 0], self.landmarks[:, 1]))

    def scale(self, s):
        m = self.get_mean()
        #centered = self.landmarks - m
        centered = self.landmarks
        points = centered.dot(s) #+ m
        return Landmark(points, self.tooth_nb)

    def rotate(self, theta):
        points = np.zeros(self.landmarks.shape)
        m = self.get_mean()
        rotation = np.array([[np.cos(theta), np.sin(theta)],
                           [-np.sin(theta), np.cos(theta)]])
        #tmp = self.landmarks - m
        tmp = self.landmarks
        for i in range(len(self.landmarks)):
            points[i, :] = tmp[i, :].dot(rotation)
        return Landmark(points, self.tooth_nb )#+ m)

    def get_landmarks(self):
        return self.landmarks

    def get_norm(self):
        return np.linalg.norm(self.landmarks)

    def translate_to_roi(self):
        x1, y1, x2, y2 = [1200, 500, 1800, 1350]
        return self.translate_axis(x1, y1)

    def translate_jaw(self, jaw_split, b_size):
        y = jaw_split - b_size
        if self.tooth_nb < 4:
            return self.translate_to_roi()
        else:
            return self.translate_to_roi().translate_axis(0, y)


    def translate_axis(self, x, y):
        landmarks = self.landmarks - [x, y]
        return Landmark(landmarks, self.tooth_nb)

    def plot(self, axis):
        if axis is None:
            _, axis = plt.subplots()
        axis.scatter(self.landmarks[:, 0], self.landmarks[:, 1])




if __name__ == '__main__':
    landmark = Landmark([13, 1], 1)
    l = landmark.to_vector()
    radiograph_path = 'Data/Radiographs/0' + str(1) + '.tif'
    rg = radiograph.Radiograph([13])

    plt.imshow(rg.img, cmap='gray')
    plt.scatter(l[:40], l[40:])
    plt.show()

    l = landmark.translate_to_roi().to_vector()
    roi = rg.roi
    plt.imshow(roi, cmap='gray')
    plt.scatter(l[:40], l[40:])
    plt.show()

    l = landmark.translate_jaw(rg.middle, rg.boundary_size).to_vector()
    plt.imshow(rg.upper, cmap='gray')
    plt.scatter(l[:40], l[40:])
    plt.show()

    landmark = Landmark([13, 6], 6)
    l = landmark.translate_jaw(rg.middle, rg.boundary_size).to_vector()
    plt.imshow(rg.lower, cmap='gray')
    plt.scatter(l[:40], l[40:])
    plt.show()




