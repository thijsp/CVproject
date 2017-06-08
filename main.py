import numpy as np
import Landmark
import DataReader
from ASM import ASM
from ProcustesAnalysis import GPA
import matplotlib.pyplot as plt
import cv2
import cv
import radiograph
import intitmanual




def click_center(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print x, y
        return np.array([[x], [y]])

if __name__ == "__main__":

    rg = radiograph.Radiograph([1])
    img, scale = rg.resize(1200, 800)
    img = img.img
    j = 3
    landmarks = []
    for i in np.arange(1, 15, 1):
        landmarks.append(Landmark.Landmark([i, j]))
    mean, shapes = GPA(landmarks)
    model = ASM(mean, shapes)

    #img = cv2.resize(img, (50, 50))
    #cv2.namedWindow('image', cv2.CV_WINDOW_AUTOSIZE)

    man = intitmanual.ManualInit(model, mean)
    man.init_manual(img)
