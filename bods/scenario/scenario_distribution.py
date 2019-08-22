# Copyright (c) 2019, Raul Astudillo

import numpy as np


class ScenarioDistribution(object):
    """
    Class to handle the scenario distribution.
    There are two possible ways to specify a parameter distribution: ...
    """

    def __init__(self, continuous=False, support=None, prob_dist=None, sample_generator=None):
        if continuous and sample_gen is None:
            pass
        else:
            self.continuous = continuous
            self.support = support
            self.prob_dist = prob_dist
            self.sample_generator = sample_generator
        
    
    def sample(self, number_of_samples=1):
        if self.support is None:
            parameter_samples = self.sample_generator(number_of_samples)
        else:
            indices = np.random.choice(int(len(self.support)), size=number_of_samples, p=self.prob_dist)
            parameter_samples = self.support[indices,:]
        return parameter_samples
