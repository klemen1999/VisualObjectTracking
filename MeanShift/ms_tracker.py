import numpy as np
from ex2_utils import Tracker, backproject_histogram, create_epanechnik_kernel, extract_histogram, get_patch
import matplotlib.pyplot as plt

class MeanShiftTracker(Tracker):
    def __init__(self, params):
        self.parameters = params

    def initialize(self, image, region):
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])

        self.kernel = create_epanechnik_kernel(self.size[1], self.size[0], self.parameters.sigma)
        self.template, _ = get_patch(image, self.position, self.kernel.shape)

        self.q = extract_histogram(self.template, self.parameters.nbins, self.kernel)
        self.q /= np.sum(self.q)

    def track(self, image):
        steps = 0
        offset_x = self.size[1]//2
        offset_y = self.size[0]//2
        column = np.arange(-offset_x, offset_x+1)
        row = np.arange(-offset_y, offset_y+1)
        [x_i, y_i] = np.meshgrid(row, column)

        p = extract_histogram(self.template, self.parameters.nbins, self.kernel)
        p /= np.sum(p)

        while steps < self.parameters.nsteps:
            v = np.sqrt(self.q / (p+self.parameters.eps))
            w = backproject_histogram(self.template, v, self.parameters.nbins)

            denominator = np.sum(w)
            if denominator == 0:
                break
            dx = np.sum(x_i*w) / denominator
            dy = np.sum(y_i*w) / denominator
            if np.linalg.norm(np.array([dx,dy])) < 0.001:
                break

            self.position =  (self.position[0]+dx, self.position[1]+dy)
            self.template, inliers  = get_patch(image, self.position, self.kernel.shape)

            p = extract_histogram(self.template, self.parameters.nbins, self.kernel)
            p /= np.sum(p)
            self.q = self.parameters.alpha*p + (1-self.parameters.alpha)*self.q
            self.q /= np.sum(self.q)
            
            steps += 1

        left = max(self.position[0]-self.kernel.shape[0]/2, 0)
        top = max(self.position[1]-self.kernel.shape[1]/2, 0)

        return [left, top, self.size[0], self.size[1]]


class MSParams():
    # Default parameter values for best overall results
    def __init__(self, sigma=1, alpha=0.000, eps=0.0005, nbins=8, nsteps=20):
        self.sigma = sigma
        self.alpha = alpha
        self.eps = eps
        self.nbins = nbins
        self.nsteps = nsteps