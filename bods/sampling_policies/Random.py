# Copyright (c) 2019, Raul Astudillo

import numpy as np
from sampling_policies.base import SamplingPolicyBase
from optimization.optimization_services import ContextManager
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.experiment_design import initial_design


class Random(SamplingPolicyBase):
    """
    Knowledge gradient acquisition function function.

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details.
    """

    analytical_gradient_prediction = True

    def __init__(self, model, optimization_space, cost_withGradients=None):
        self.decision_context_space = optimization_space.decision_context_space
        super(Random, self).__init__(model, optimization_space, cost_withGradients=cost_withGradients)
        if cost_withGradients is None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            self.cost_withGradients = cost_withGradients
            
    def suggest_sample(self, number_of_samples=1):
        """
        Returns a suggested next point to evaluate.
        """
        suggested_sample = initial_design('random', self.decision_context_space, 1)
        return suggested_sample