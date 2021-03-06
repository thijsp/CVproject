import numpy as np
import matplotlib.pyplot as plt
import preprocessing


def get_derivates(a, centre, img, nb_pixels):
    img_sobel = preprocessing.togradient_sobel(img)
    intens = get_intensities(a, centre, img, nb_pixels)[1:-1, :]
    values = get_intensities(a, centre, img_sobel, nb_pixels)[1:-1, :]
    ins =  values[:, 2]
    ins[ins == 0] = 1
    fac = np.sum(np.abs(intens[:, 2]))
    for i in range(len(ins)):
        values[i, 2] = ins[i] * 1.0 / fac
    return values




def get_derivates_old(a, centre, img, nb_pixels):
    values = get_intensities(a, centre, img, nb_pixels)
    ins = values[:, 2]
    #g = ins[1:-1]
    ins[ins==0] = 1 # Avoid dividing by zero
    d1 = (ins[1:]-ins[:-1]) / ins[:-1]    # Percentage change in intensity => gradient to the right
    d2 = (ins[:-1] - ins[1:]) / ins[1:]   # Percentage change in intensity => gradient to the left
    g = d1[1:] + d2[:-1] # Added sum instead of minus as gradient vectors should be added up to get the resulting gradient vector. Indexes were added to ensure gradients related to the same pixel are added.
    if np.isnan(g).any():
        pass
    g = g / np.sum(np.abs(g))
    #norm = np.sum(np.abs(g))
    #g = g / norm
    values = values[1:-1]
    values[:, 2] = g
    return values


def get_intensities(a_normal, centre, img, nb_pixels):
    x_c, y_c = centre
    a = a_normal
    theta = np.arctan(a)#+np.pi/2
    dist = 3.0*nb_pixels
    x_d = np.cos(theta) * dist
    y_d = np.sin(theta) * dist
    x_1 = x_d + x_c
    y_1 = y_d + y_c
    x_2 = x_c - x_d
    y_2 = y_c - y_d

    if x_1 > x_2:
        x_1, y_1, x_2, y_2 = x_2, y_2, x_1, y_1

    #intensities = createLineIterator([x_1, y_1], [x_2, y_2], img)
    intensities1 = createLineIterator(centre, [x_1, y_1], img)
    intensities2 = createLineIterator(centre, [x_2, y_2], img)
    ins1 = filter(intensities1, nb_pixels+1)
    ins2 = filter(intensities2, nb_pixels+1)
    ins1 = arrange(ins1, a)
    ins2 = arrange(ins2, a)

    ins_c = np.array([[centre[0], centre[1], img[int(centre[1]), int(centre[0])]]])
    ins = np.vstack((ins1, ins_c, ins2))
    return ins

def filter(intensities, nb_pixels):
    nb_del = len(intensities) - nb_pixels
    del_s = int(nb_del)
    return intensities[:-del_s]

def arrange(intensities, a_normal): # Arrange array so that the middle pixel (next to the center) does not get deleted in the calculation of gradients
    if intensities.size == 0:
        pass
    if a_normal == float('inf'):
        if intensities[0, 1] > intensities[-1, 1]:
            intensities = intensities[::-1]
    else:
        if intensities[0, 0] > intensities[-1, 0]:
            intensities = intensities[::-1]

    return intensities


# def filter(intensities, nb_pixels):
#     nb_del = len(intensities) - nb_pixels
#     del_s = int(nb_del/2)
#     ins = intensities[del_s:-del_s, :]
#     if len(ins) > nb_pixels:
#         ins = ins[:-1]
#     return ins


def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
   #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = int(P1[0])
    P1Y = int(P1[1])
    P2X = int(P2[0])
    P2Y = int(P2[1])

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)
    #dXa = 25
    #dYa = 25

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dXa, dYa), 3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1, 1)
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1, 1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = float(dX)/float(dY)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1, 1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = float(dY)/float(dX)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1, 1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]

    img_in = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]
    img_out = itbuffer[(colX < 0) | (colY < 0) | (colX > imageW) | (colY > imageH)]
    #Get intensities from img ndarray
    img_in[:, 2] = img[img_in[:,1].astype(np.uint),img_in[:,0].astype(np.uint)]
    img_out[:, 2] = 0

    if img_out.size > 0:
        itbuffer = np.vstack((img_in, img_out))
        itbuffer = itbuffer[itbuffer[:, 0].argsort()]
    else:
        itbuffer = img_in

    return itbuffer

def get_normal(landmark, l):
    if l == 0:
        p1 = landmark.landmarks[-1]
    else:
        p1 = landmark.landmarks[l - 1]
    if l == len(landmark.landmarks) - 1:
        p2 = landmark.landmarks[0]
    else:
        p2 = landmark.landmarks[l + 1]
    x1, y1 = p1
    x2, y2 = p2

    # y = a*x + b -> y = y1 + (y2 - y1)/(x2 - x1) * (x - x1)
    a = (y2 - y1) / (x2 - x1)
    # b = -a*x1 + y1

    # normal a -> -a, through l
    x, y = landmark.landmarks[l]
    if np.abs(a) > 3.5 or (x2 - x1) == 0: #10 ** 10:
        #a = a
        #a = float('inf')
        a = 0
        # no b
        b = 0
    elif np.abs(a) < 0.6: #10 ** (-10):
        #a = float('inf')
        #a = 0
        a = float('inf')
        b = y1
    else:
        a = -1.0 / a
        b = y - a * x

    return a, b


def get_coordinates(a, b, middle, nb_pixels):
    nb_samples = nb_pixels / 2
    x_m = middle[0]
    x_0 = x_m - nb_samples
    x_1 = x_m + nb_samples
    # y = a*x + b
    xs = np.arange(x_0, x_1, 1)
    if a == 0:
        x = xs
        y = np.array([b for _ in np.arange(0, nb_samples * 2, 1)])
    elif a == float('inf'):
        x = np.array([middle[0] for _ in np.arange(0, nb_samples * 2, 1)])
        y = np.array([middle[1] + i for i in np.arange(-nb_samples, nb_samples, 1)])
    else:
        x = xs
        y = np.array([a * x_i + b for x_i in xs])
    return np.array([x, y])


def get_line_intensities(coordinates, img):
    return img[coordinates[:, 1], coordinates[:, 1]]
