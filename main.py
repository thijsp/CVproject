import numpy as np
import Landmark
from ASM import ASM
from ProcustesAnalysis import GPA
import matplotlib.pyplot as plt



if __name__ == "__main__":
    j = 3
    landmarks = []
    for i in np.arange(1, 15, 1):
        landmarks.append(Landmark.Landmark([i, j]))
    mean, shapes = GPA(landmarks)
    model = ASM(mean, shapes)
    model.fit()
