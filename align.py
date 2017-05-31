from __future__ import division
import DataReader
import Aligner
import Plot
import numpy as np
import cv2



"""
Lecture: Active Shape Models
https://www.youtube.com/watch?v=53kx_czs7Es
"""

if __name__ == '__main__':
    # Load Data
    landmarks = DataReader.read_landmarks()

    # plot the radiograph with landmarks on it
    Plot.plot_radiographs(landmarks)

    # Procrustes Analysis
    landmarks_aligned = Aligner.procrustes_analysis(landmarks)

    # PCA
    eigs, eigv = Aligner.perform_pca(landmarks_aligned)

    a = [[1250, 700, 1750, 1300],
         [1300, 650, 1750, 1300],
         [1300, 700, 1800, 1350],
         [1300, 625, 1700, 1325],
         [1350, 700, 1800, 1300],
         [1300, 600, 1800, 1250],
         [1300, 650, 1725, 1300],
         [1350, 750, 1725, 700],
         [1300, 700, 1750, 1350],
         [1300, 500, 1700, 1250],
         [1200, 800, 1700, 1350],
         [1300, 700, 1700, 1150],
         [1350, 500, 1750, 1200],
         [1200, 630, 1700, 1200]]


[ 1292.85714286,   657.5       ,  1739.28571429,  1237.5       ]
