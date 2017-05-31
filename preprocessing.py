from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimpg
from scipy import signal
import cv2

"""Implementation of filters to preprocess the radiographs
    http://www.ijcst.com/vol24/2/vijaykumari.pdf
    butterworth: https://books.google.be/books?id=9irSCgAAQBAJ&pg=PA182&lpg=PA182&dq=scipy+ndimage+butterworth&source=bl&ots=YLIm5AFeG6&sig=kzYNvZUmblKbNF8xt4N2tU-rl3o&hl=en&sa=X&ved=0ahUKEwjD2tGhgdvTAhVBJlAKHc3OAVAQ6AEIPjAF#v=onepage&q=scipy%20ndimage%20butterworth&f=false
    median filter: http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#medianblur
    bilateral filter: http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#bilateralfilter
"""

def get_roi(img):
    roi = cv2.rectangle()


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
    img_blur = cv2.MedianBlur(img, par)
    return img_blur

def gauss_filter(img, par):
    img_blur = cv2.GaussianBlur(img, ksize=(0,0), sigmaX=4, sigmaY=8)
    return img_blur


def bilateral_filter(img):
    img_bi = cv2.bilateralFilter(src=img, d=15, sigmaColor=300, sigmaSpace=300)
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
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 2)



if __name__ == '__main__':
    img = cv2.imread('Data/Radiographs/01.tif')
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print img_grey.shape
    img_roi = get_roi(img_grey)

    print img_roi.shape

    print np.mean(np.std(img_roi, axis=0))
    print np.mean(np.std(img_roi, axis=1))

    img_med = gauss_filter(img_roi, 21)
    plt.imshow(img_med, cmap='gray')
    plt.show()
    # plt.subplot(211)
    # plt.imshow(img_grey, cmap='gray')
    # plt.subplot(212)
    # plt.imshow(img_med, cmap='gray')
    # plt.show()

    #img_can = canny(img_med, 15, 40)
    img_can = adaptive_threshold(img_med)
    plt.imshow(img_can, cmap='gray')
    plt.show()

    # t = [25, 30, 35, 40, 50, 60, 70]
    # for i in range(len(t)):
    #     img_can = canny(img_grey, t[i], 2)
    #     plt.imshow(img_can, cmap='gray')
    #     plt.show()


    #fshift = get_dft(img)
    #magnitude_spectrum = 20 * np.log(np.abs(fshift))
    #plt.subplot(121), plt.imshow(img, cmap='gray')
    #plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    #plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    #plt.show()