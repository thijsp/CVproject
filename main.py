import numpy as np
import Landmark
import DataReader
from ASM import ASM
from ProcustesAnalysis import GPA
import matplotlib.pyplot as plt
import cv2
import cv




def click_center(event, x, y, flags, param):
    return np.array([[x], [y]])

if __name__ == "__main__":
    img = DataReader.read_radiograph(1)
    j = 3
    landmarks = []
    for i in np.arange(1, 15, 1):
        landmarks.append(Landmark.Landmark([i, j]))
    mean, shapes = GPA(landmarks)
    model = ASM(mean, shapes)

    #img = cv2.resize(img, (50, 50))
    #cv2.namedWindow('image', cv2.CV_WINDOW_AUTOSIZE)

    cv2.imshow('choose', img)
    cv2.resizeWindow('image', 60,60)

    cv.SetMouseCallback('choose', click_center)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    model.fit_manual()
