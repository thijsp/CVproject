import matplotlib.pyplot as plt
import matplotlib.image as mimpg
import seaborn as sns
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