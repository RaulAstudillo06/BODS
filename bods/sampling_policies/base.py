# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPyOpt.core.task.cost import constant_cost_withGradients

class SamplingPolicyBase(object):
    """
    Base class for sampling in Bayesian Optimization

    :param model: GPyOpt class of model
    :param optimization_space:
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer

    """

    analytical_gradient_prediction = False

    def __init__(self, model, optimization_space, cost_withGradients=None):
        self.model = model
        self.optimization_space = optimization_space
        self.analytical_gradient_objective = self.analytical_gradient_prediction and self.model.analytical_gradient_prediction # flag from the model to test if gradients are available

        if cost_withGradients is  None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            self.cost_withGradients = cost_withGradients

    @staticmethod
    def fromDict(model, space, optimizer, cost_withGradients, config):
        raise NotImplementedError()

    def suggest_sample(self, number_of_samples=1):
        """
        Returns a suggested next point to evaluate.
        """
        raise NotImplementedError('')