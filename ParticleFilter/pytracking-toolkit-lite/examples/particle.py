import numpy as np
import random

from utils.tracker import Tracker
from utils.my_utils import get_patch, create_epanechnik_kernel, extract_histogram, kalman_init, sample_gauss

# Best config:
# self.sigma = 0.05
# self.nbins = 8
# self.model = "RW"
# self.N = 100
# self.alpha = 0.01
# sigma epinechink 0.5
# On chosen 5 sequences: Overlap: 0.48, Number of fails: 6

class ParticleTracker(Tracker):
    def __init__(self):
        self.sigma = 0.05
        self.nbins = 8
        self.model = "RW"
        self.N = 80
        self.alpha = 0.01
        random.seed(42)
        np.random.seed(42)


    def name(self):
        return "particle"


    def initialize(self, image, region):
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])

        self.kernel = create_epanechnik_kernel(self.size[1], self.size[0], 0.5)
        template, _ = get_patch(image, self.position, self.kernel.shape)

        self.hist = extract_histogram(template, self.nbins, self.kernel)
        self.hist /= np.sum(self.hist)

        q = 0.1 * min(self.size)

        if self.model == "RW":
            self.position = np.array([self.position[0], self.position[1]])
        elif self.model == "NCV":
            self.position = np.array([self.position[0], self.position[1], 0, 0])
        elif self.model == "NCA":
            self.position = np.array([self.position[0], self.position[1], 0, 0, 0, 0])
        else:
            raise Exception("Model name not recognized")
        
        self.A, _, self.Q, _ = kalman_init(self.model, q, 0)
        
        self.particles = np.ones((self.N,self.position.shape[0])) * self.position + sample_gauss(np.zeros(self.Q.shape[0]), self.Q, self.N)
        self.weights = np.ones(self.N) / self.N
        

    def resample_particles(self):
        weights_norm = self.weights / np.sum(self.weights)
        weights_cumsumed = np.cumsum(weights_norm)
        rand_samples = np.random.rand(self.N , 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        self.particles = self.particles[sampled_idxs.flatten(), :]


    def calc_similarity(self, hist1, hist2):
        # Hellinger distance^2
        return 1 - np.dot(np.sqrt(hist1),np.sqrt(hist2))


    def track(self, image):
        self.resample_particles()
        self.weights = np.ones(self.N) / self.N

        for i in range(self.N):
            self.particles[i,:] = self.particles[i,:] + sample_gauss(np.zeros(self.Q.shape[0]), self.Q, 1)
            template, _ = get_patch(image, self.particles[i,:], self.kernel.shape)
            curr_hist = extract_histogram(template, self.nbins, self.kernel)
            curr_hist /= np.sum(curr_hist)
            similarity = self.calc_similarity(self.hist, curr_hist)
            self.weights[i] = np.exp(-0.5*similarity/(self.sigma**2))
        
        self.weights /= np.sum(self.weights)
        self.position = np.sum(self.particles * self.weights[:,np.newaxis], axis=0)
        template, _ = get_patch(image, self.position, self.kernel.shape)
        new_hist = extract_histogram(template, self.nbins, self.kernel)
        new_hist /= np.sum(new_hist)

        self.hist = (1-self.alpha)*self.hist + self.alpha*new_hist
        
        left = max(self.position[0]-self.kernel.shape[0]/2, 0)
        top = max(self.position[1]-self.kernel.shape[1]/2, 0)
        
        return [left, top, self.size[0], self.size[1]]

    
    def get_current_particles(self):
        return self.particles, self.weights