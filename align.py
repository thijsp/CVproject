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

    sort_args = np.fliplr(np.argsort(eigs))
    sort_eigs = eigs[sort_args]
    print sort_eigs

