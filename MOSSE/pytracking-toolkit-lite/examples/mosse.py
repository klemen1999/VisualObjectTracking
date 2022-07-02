import cv2
import numpy as np
import random

from utils.tracker import Tracker
from utils.ex3_utils import get_patch, create_gauss_peak, create_cosine_window

"""
Best configs:
mosse_basic: sigma=2, alpa=0.125, lambd=1e-3
mosse_enlarge: sigma=2, alpa=0.125, lambd=1e-3, enlarge_factor=1.75
mosse_improved: sigma=2, alpa=0.15, lambd=1e-3, enlarge_factor=1.75
"""

class MOSSETracker(Tracker):
    def __init__(self):
        self.sigma = 2
        # self.alpha = 0.125
        self.alpha = 0.15
        self.lambd = 1e-3
        # self.enlarge_factor = None
        self.enlarge_factor = 1.75
        self.use_improved = True
        random.seed(42)

    def name(self):
        if self.enlarge_factor is None and not self.use_improved:
            return "mosse_basic"
        elif not self.use_improved:
            return "mosse_enlarge"
        else:
            return 'mosse_improved'

    def initialize(self, image, region):
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])

        if self.enlarge_factor is not None:
            self.window = (self.size[0]*self.enlarge_factor, self.size[1]*self.enlarge_factor)
        else:
            self.window = self.size

        self.G = create_gauss_peak(self.window, self.sigma)
        self.G_fft = np.fft.fft2(self.G)

        self.cosine = create_cosine_window((self.G.shape[1], self.G.shape[0]))
        F, inliers = get_patch(image, self.position, (self.G.shape[1], self.G.shape[0]))
        
        if self.use_improved:
            self.A = np.zeros_like(self.G_fft)
            self.B = np.zeros_like(self.G_fft)+self.lambd
            for _ in range(8):
                Fi = self.random_perturbation(F)
                Fi = Fi * self.cosine * inliers
                Fi_fft = np.fft.fft2(Fi)
                Fi_conj = np.conjugate(Fi_fft)
                self.A += self.G_fft * Fi_conj
                self.B +=  Fi_fft * Fi_conj
        else:
            F = F * self.cosine * inliers
            F_fft = np.fft.fft2(F)
            F_conj = np.conjugate(F_fft)
            self.H = (self.G_fft * F_conj) / \
                (F_fft * F_conj + self.lambd)

    def random_perturbation(self, img):
        # Random rotation from -10 to 10 degrees
        angle = random.uniform(-10,10)
        rot_mat = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2), angle, 1)
        img_rotated = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
        return img_rotated

    def track(self, image):
        F, inliers = get_patch(image, self.position, (self.G.shape[1], self.G.shape[0]))
        F = F * self.cosine * inliers
        F_fft = np.fft.fft2(F)

        if self.use_improved:
            H = self.A / self.B
            R = np.fft.ifft2(F_fft * H)
        else:
            R = np.fft.ifft2(F_fft * self.H)

        height, width = R.shape
        y_new, x_new = np.unravel_index(R.argmax(), R.shape)
        if x_new > width / 2:
            x_new = x_new - width
        if y_new > height / 2:
            y_new = y_new - height
        self.position = (self.position[0]+x_new, self.position[1]+y_new)
        
        F, inliers = get_patch(image, self.position, (self.G.shape[1], self.G.shape[0]))
        F = F * self.cosine * inliers

        F_fft = np.fft.fft2(F)
        F_conj = np.conjugate(F_fft)
        
        if self.use_improved:
            self.A = self.alpha*(self.G_fft*F_conj) + (1-self.alpha)*self.A
            self.B = self.alpha*(F_fft*F_conj+self.lambd) + (1-self.alpha)*self.B
        else:
            H_new = (self.G_fft * F_conj) / \
                (F_fft * F_conj + self.lambd) 
            self.H = (1-self.alpha)*self.H + self.alpha*H_new

        left = max(self.position[0]-self.size[0]/2, 0)
        top = max(self.position[1]-self.size[1]/2, 0)

        return [left, top, self.size[0], self.size[1]]