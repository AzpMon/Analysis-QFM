from scipy.stats import rv_continuous
import numpy as np


### Class created for the use of sampling distributions
class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)