import numpy as np
import cv2
import matplotlib.pyplot as plt
import Landmark
import radiograph
import ASM
import intitmanual
import ProcustesAnalysis
import intitmanual


class Fitter(object):

    def __init__(self, asm, radiograph):
        self.rg = radiograph
        self.asm = asm

    def fit_new(self):
        nb_comp = len(self.asm.eigvals)
        b = np.zeros((nb_comp, 1))
        manual = intitmanual.ManualInit(self.asm)
        t, s, theta, landmark = manual.init_manual(self.rg)
        estimate = landmark
        while True:
            next_landmark = self.asm.search_profile(estimate, self.rg)
            t, s, theta, b_next = self.calculate_b(next_landmark)

            if (b_next - b < 10 ** (-8)).all():
                b = b_next
                break
            else:
                b = b_next
                estimate = Landmark.Landmark(self.asm.reconstruct(b)[0], self.asm.tooth_nb)
                estimate_new = self.project(t, s, theta, estimate)
                _, ax = plt.subplots()
                ax.imshow(self.rg.upper, cmap='gray')
                estimate.plot(ax)
                estimate_new.plot(ax)
                plt.show()
                estimate = estimate_new
        return b

    def calculate_b(self, estimate):
        estimate_org = estimate.transform_origin()
        t, s, theta = ProcustesAnalysis.get_parameters(estimate_org, self.asm.mean_shape)
        proj_landmark = self.project(t, s, theta, estimate_org)
        tangent_landmark = self.tangent(proj_landmark, self.asm.mean_shape)
        b_next = self.asm.estimate_model_params(tangent_landmark)
        t, s, theta = ProcustesAnalysis.get_parameters(self.asm.mean_shape, estimate_org)
        t = estimate.get_center()
        return t, s, theta, b_next

    def project(self, t, s, theta, estimate):
        estimate = estimate.rotate(theta)
        estimate = estimate.scale(s)
        estimate = estimate.transform_to_point(t)
        #scale_fac = np.dot(estimate.to_vector(), self.asm.mean_shape.to_vector())
        vec = estimate.to_vector()
        return Landmark.Landmark(vec, estimate.tooth_nb)

    def tangent(self, estimate, mean_shape):
        scale_fac = np.dot(estimate.to_vector(), mean_shape.to_vector())
        vec = estimate.to_vector()
        return Landmark.Landmark(vec * 1.0/scale_fac, estimate.tooth_nb)


if __name__ == '__main__':
    j = 1
    landmarks = []
    rgs = []
    for i in np.arange(1, 15, 1):
        rgs.append(radiograph.Radiograph([i]))
        landmarks.append(Landmark.Landmark(mark=[i, j], tooth_nb=j))
    asm = ASM.ASM(landmarks, rgs)
    est = asm.fit(rgs[0], False)
    fitter = Fitter(asm, rgs[0])
    fitter.fit_new()
