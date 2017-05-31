import matplotlib.pyplot as plt
import matplotlib.image as mimpg
import seaborn as sns
import numpy as np
import cv2
plt.style.use(['seaborn-white','seaborn-paper'])
sns.set(font='serif')


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


def plot_radiographs(landmarks):
    for i in range(1, 15):
        if i < 10:
            radiograph_path = 'Data/Radiographs/0' + str(i) + '.tif'
        else:
            radiograph_path = 'Data/Radiographs/' + str(i) + '.tif'
        plot_radiograph(radiograph_path, landmarks[:, i - 1, :, :])

def plot_roi(radiograph_path, ptx, pty):
    img = mimpg.imread(radiograph_path)
    cv2.rectangle(img, ptx, pty, color=200, thickness=3)
    plt.imshow(img)
    plt.show()

def define_roi():
    x1, y1, x2, y2 = [1200,  500, 1800,  1350]
    for i in range(1, 15):
        if i < 10:
            radiograph_path = 'Data/Radiographs/0' + str(i) + '.tif'
        else:
            radiograph_path = 'Data/Radiographs/' + str(i) + '.tif'
        plot_roi(radiograph_path, (x1,y1), (x2,y2))


if __name__ == '__main__':
    define_roi()
    x1, y1, x2, y2 = [1292.85714286, 657.5, 1739.28571429, 1237.5]