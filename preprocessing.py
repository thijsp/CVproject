from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimpg
from scipy import signal
import cv2
from scipy.ndimage import morphology

"""Implementation of filters to preprocess the radiographs
    http://www.ijcst.com/vol24/2/vijaykumari.pdf
    butterworth: https://books.google.be/books?id=9irSCgAAQBAJ&pg=PA182&lpg=PA182&dq=scipy+ndimage+butterworth&source=bl&ots=YLIm5AFeG6&sig=kzYNvZUmblKbNF8xt4N2tU-rl3o&hl=en&sa=X&ved=0ahUKEwjD2tGhgdvTAhVBJlAKHc3OAVAQ6AEIPjAF#v=onepage&q=scipy%20ndimage%20butterworth&f=false
    median filter: http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#medianblur
    bilateral filter: http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#bilateralfilter
"""


def butterworth_lowpass(cutoff_freq=15, order=2, size=256):
    ar = np.arange(size//2, size//2, 1.0)
    x, y = np.meshgrid(ar, ar)
    bl = 1 / (1 + np.power((x**2 + y**2) / cutoff_freq**2, order))
    return bl


def butterworth_highpass(cutoff_freq=15, order=2, size=256):
    bl = 1 - butterworth_lowpass(cutoff_freq, order, size)
    return bl


def get_dft(img):
    return np.fft.fftshift(np.fft.fft2(img))

def inv_dft(f):
    return np.fft.ifft2(np.fft.ifftshift(f))


def median_filter(img, par):
    img_blur = cv2.medianBlur(img, par)
    return img_blur

def gauss_filter(img):
    img_blur = cv2.GaussianBlur(img, ksize=(0,0), sigmaX=4, sigmaY=8)
    return img_blur


def bilateral_filter(img):
    img_bi = cv2.bilateralFilter(src=img, d=9, sigmaColor=175, sigmaSpace=175)
    return img_bi


def scharr(img):
    return cv2.Scharr(img, ddepth=-1, dx=1, dy=0)


def canny(img, low_thresh, high_thresh):
    return cv2.Canny(img, low_thresh, high_thresh)


def get_roi(img):
    x1, y1, x2, y2 = [1200,  500, 1800,  1350]
    img = img[y1:y2, x1:x2]
    return img


def adaptive_threshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 0.7)


def sobel(img):
    return cv2.Sobel(img, ddepth=-1, ksize=3, dx=2, dy=1)


def histogram_eq(img):
    return cv2.equalizeHist(img)


def level_down(img):
    return cv2.pyrDown(img)

def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(16, 16))
    return clahe.apply(img)


def gaussian_pyramid(img):
    G = img.copy()
    gpA = [G]
    for k in xrange(6):
        G = cv2.pyrDown(gpA[k])
        gpA.append(G)
    return gpA


def laplacian_pyramid(img):
    gpA = gaussian_pyramid(img)
    lpA = [gpA[5]]
    for i in xrange(5, 0, -1):
        size = (gpA[i - 1].shape[1], gpA[i - 1].shape[0])
        GE = cv2.pyrUp(gpA[i], dstsize=size)
        L = cv2.subtract(gpA[i-1], GE)
        lpA.append(L)
    return lpA


def reconstruct_laplacian(LS):
    ls_ = LS[0]
    for i in xrange(1, 6):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = cv2.add(ls_, LS[i])
    return ls_


def preprocess(img):
    img = bilateral_filter(img)
    #img = gauss_filter(img)
    #LS = laplacian_pyramid(img)
    #LS[3][LS[3] > 1] = 255
    #img = scharr(LS[3])

    img_top = morphology.white_tophat(img, size=100)
    img_bottom = morphology.black_tophat(img, size=80)
    #LS[1] = bilateral_filter(LS[1])
    # LS[2] = histogram_eq(LS[2])
    #LS[3] = histogram_eq(LS[3])

    img = cv2.add(img, img_top)
    img = cv2.subtract(img, img_bottom)

    img = clahe(img)
    #img = togradient_sobel(img)

    #img = canny(img, 50, 60)

    #LS = laplacian_pyramid(img)
    #img = reconstruct_laplacian(LS)
    # LS = laplacian_pyramid(img)
    # img = LS[3]
    return img

def split_processing(img):
    LS = laplacian_pyramid(img)

    LS[1] = bilateral_filter(LS[1])
    # LS[2] = histogram_eq(LS[2])
    LS[3] = histogram_eq(LS[3])

    img = reconstruct_laplacian(LS)
    # LS = laplacian_pyramid(img)
    # img = LS[3]
    return img

def togradient_sobel(img):
    """Applies the Sobel Operator.
    Args:
        img: A grayscale dental x-ray image.
    Returns:
        An image with the detected edges bright on a darker background.
    """
    img = cv2.GaussianBlur(img,(3,3),0)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)




if __name__ == '__main__':
    img = cv2.imread('Data/Radiographs/01.tif')
    i = get_roi(img)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img_grey

    img_grey = get_roi(img_grey)

    img = median_filter(img_grey, 7)

    LS = laplacian_pyramid(img)

    plt.imshow(LS[3], cmap='gray')
    plt.show()


    LS[1] = bilateral_filter(LS[1], 17)
    #LS[2] = histogram_eq(LS[2])
    LS[3] = histogram_eq(LS[3])
    


    img = reconstruct_laplacian(LS)
    LS = laplacian_pyramid(img)

    img = LS[3]


    plt.imshow(img, cmap='gray')
    plt.show()
