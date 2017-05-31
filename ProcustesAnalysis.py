import numpy as np
import Landmark
import matplotlib.pyplot as plt

def GPA(landmarks):
    shapes = list(landmarks)
    shapes = [landmark.transform_origin() for landmark in shapes]
    mean0 = shapes[0].transform_unit()
    mean_shape = mean0
    while True:
        for i, shape in enumerate(shapes):
            shapes[i] = align(shape, mean_shape)
        next_mean = get_mean_shape(shapes)
        next_mean = align(next_mean, mean0)
        next_mean_scaled = next_mean.transform_unit()
        next_mean = next_mean_scaled.transform_origin()

        if ((next_mean.to_vector() - mean_shape.to_vector()) < 1e-10).all():
            break
        mean_shape = next_mean

    return mean_shape, shapes


def align(example, mean_shape):
    t, s, theta = get_parameters(example, mean_shape)

    example = example.rotate(theta)
    example = example.scale(s)
    scale_fac = np.dot(example.to_vector(), mean_shape.to_vector())
    vec = example.to_vector()
    return Landmark.Landmark(vec*(1.0/scale_fac))


def get_parameters(example, mean_shape):
    example = example.to_vector()
    mean_shape = mean_shape.to_vector()

    x_length = len(mean_shape)//2

    cen_mean = [np.mean(mean_shape[:x_length]), np.mean(mean_shape[x_length:])]
    ex_mean = [np.mean(example[:x_length]), np.mean(example[x_length:])]
    cen_mean = np.array(cen_mean)
    ex_mean = np.array(ex_mean)

    ex = [x - ex_mean[0] for x in example[:x_length]] + [y - ex_mean[1] for y in example[x_length:]]
    m = [x - cen_mean[0] for x in mean_shape[:x_length]] + [y - cen_mean[1] for y in mean_shape[x_length:]]

    norm_ex = (np.linalg.norm(ex)**2)

    a = np.dot(ex, m) / norm_ex
    b = (np.dot(ex[:x_length], m[x_length:]) - np.dot(ex[x_length:], m[:x_length])) / norm_ex
    s = np.sqrt(a**2 + b**2)
    theta = np.arctan(b/a)
    t = cen_mean - ex_mean

    return t, s, theta


def get_mean_shape(shapes):
    landmark_vec = []
    for i, shape in enumerate(shapes):
        landmark_vec.append(shape.to_vector())
    landmark_vec = np.array(landmark_vec)
    m = np.mean(landmark_vec, axis=0)
    return Landmark.Landmark(m)

if __name__ == '__main__':
    j = 3
    landmarks = []
    for i in np.arange(1, 15, 1):
        landmarks.append(Landmark.Landmark([i, j]))
    mean, shapes = GPA(landmarks)
    mean = mean.to_vector()
    plt.figure()
    plt.scatter(mean[:40], mean[40:])
    plt.show()
