# Copyright (c) 2019, Raul Astudillo

import numpy as np
from scipy.spatial.distance import euclidean
from sampling_policies.base import SamplingPolicyBase
from optimization.optimization_services import ContextManager
from GPyOpt.core.task.cost import constant_cost_withGradients
from GPyOpt.experiment_design import initial_design
from GPyOpt.optimization.optimizer import apply_optimizer, choose_optimizer


class TS(SamplingPolicyBase):
    """
    Knowledge gradient acquisition function function.

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details.
    """

    analytical_gradient_prediction = True

    def __init__(self, model, optimization_space, optimizer, scenario_distribution, utility, expectation_utility, cost_withGradients=None):
        self.optimizer_name = optimizer
        self.scenario_distribution = scenario_distribution
        self.utility = utility
        self.expectation_utility = expectation_utility
        super(TS, self).__init__(model, optimization_space, cost_withGradients=cost_withGradients)
        if cost_withGradients is None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            self.cost_withGradients = cost_withGradients
        #
        self.decision_space = optimization_space.decision_space
        self.decision_space_context_manager = ContextManager(self.decision_space)
        self.decision_space_optimizer = choose_optimizer(self.optimizer_name, self.decision_space_context_manager.noncontext_bounds)
        #
        self.decision_context_space = optimization_space.decision_context_space
        self.decision_context_space_context_manager = ContextManager(self.decision_context_space)
        self.decision_context_space_optimizer = choose_optimizer(self.optimizer_name, self.decision_context_space_context_manager.noncontext_bounds)
        #
        self.utility_prob_dist = self.utility.parameter_distribution
        self.full_scenario_support = True
        if self.full_scenario_support:
            self.scenario_support = scenario_distribution.support
            self.scenario_prob_dist = scenario_distribution.prob_dist
            self.scenario_support_cardinality = len(self.scenario_support)
        self.utility_support = utility.parameter_distribution.support
        self.utility_prob_dist = utility.parameter_distribution.prob_dist
        self.full_utility_support = self.utility.parameter_distribution.use_full_support
        if self.full_utility_support:
            self.utility_support_cardinality = len(self.utility_support)
        self.number_of_gp_hyps_samples = min(10, self.model.number_of_hyps_samples())
        self.X_aux = None
        self.Y_aux = None
            
    def suggest_sample(self, number_of_samples=1):
        """
        Returns a suggested next point to evaluate.
        """
        utility_parameter_sample = self.utility.sample_parameter(number_of_samples=1)
        model_sample = self.model.get_copy_of_model_sample()
        X_evaluated = np.copy(model_sample.X)
        Y_evaluated = np.copy(model_sample.Y)
        self.X_aux = np.copy(X_evaluated)
        self.Y_aux = np.copy(Y_evaluated)
        
        def objective_func_sample(d):
            X_new = np.vstack([np.append(d, theta) for theta in self.scenario_support])
            Y_new = model_sample.posterior_samples_f(X_new, size=1, full_cov=True)
            self.X_aux = np.vstack((self.X_aux, X_new))
            self.Y_aux = np.vstack((self.Y_aux, Y_new))
            model_sample.set_XY(self.X_aux, self.Y_aux)
            val = 0.
            for w in range(self.scenario_support_cardinality):
                    val += self.scenario_prob_dist[w]*self.utility.eval_func(Y_new[w, 0], utility_parameter_sample)
            return -val
        
        d0 = initial_design('random', self.decision_space, 1)
        try:
            #argmax =  self.decision_space_optimizer.optimize(d0, objective_func_sample, maxfevals=200)[0]
            argmax = apply_optimizer(self.decision_space_optimizer, d0, f=objective_func_sample, context_manager=self.decision_space_context_manager, space=self.decision_space, maxfevals=200)[0]
        except:
            argmax = d0

        aux_grid =  np.vstack([np.append(argmax, theta) for theta in self.scenario_support])
        self.model.set_hyperparameters(0)
        var = self.model.posterior_variance(aux_grid)
        for h in range(1, self.number_of_gp_hyps_samples):
            self.model.set_hyperparameters(h)
            var += self.model.posterior_variance(aux_grid)
        var = var[:, 0]
        index =  np.argmax(var)
        suggested_sample = np.append(argmax, self.scenario_support[index])
    
        use_suggested_sample = True
        i = 0
        min_distance = np.infty
        while use_suggested_sample and i < X_evaluated.shape[0]:
            distance_to_evaluated_point = euclidean(X_evaluated[i, :], suggested_sample)
            if distance_to_evaluated_point < min_distance:
                min_distance = distance_to_evaluated_point
            if distance_to_evaluated_point < 1e-1/np.sqrt(X_evaluated.shape[1]):
                use_suggested_sample = False
            i += 1
        
        print('Minimum distance to previously evaluated point is: {}'.format(min_distance))
        
        if not use_suggested_sample:
            print('Suggested point is to close to previously evaluated point; swithching to max expected value sampling policy.')
            
            def expectation_objective_func(d):
                d = np.atleast_2d(d)
                func_val = 0.
                cross_product_grid = np.vstack([np.append(d, theta) for theta in self.scenario_support])
                for h in range(self.number_of_gp_hyps_samples):
                    self.model.set_hyperparameters(h)
                    mean, var = self.model.predict_noiseless(cross_product_grid) 
                    for w in range(self.scenario_support_cardinality):
                        expectation_utility = self.expectation_utility.eval_func(mean[w,0], var[w,0], utility_parameter_sample)
                        func_val += self.scenario_prob_dist[w] * expectation_utility
                func_val /= self.number_of_gp_hyps_samples
                func_val = func_val[:, 0]
                return -func_val         
            
            #argmax =  self.decision_space_optimizer.optimize(d0, expectation_objective_func)[0]
            argmax = apply_optimizer(self.decision_space_optimizer, d0, f=expectation_objective_func, context_manager=self.decision_space_context_manager, space=self.decision_space)[0]
            
            aux_grid =  np.vstack([np.append(argmax, theta) for theta in self.scenario_support])
            self.model.set_hyperparameters(0)
            var = self.model.posterior_variance(aux_grid)
            for h in range(1, self.number_of_gp_hyps_samples):
                self.model.set_hyperparameters(h)
                var += self.model.posterior_variance(aux_grid)
            var = var[:, 0]
            index =  np.argmax(var)
            suggested_sample = np.append(argmax, self.scenario_support[index])            
            use_suggested_sample = True
            i = 0
            min_distance = np.infty
            while use_suggested_sample and i < X_evaluated.shape[0]:
                distance_to_evaluated_point = euclidean(X_evaluated[i, :], suggested_sample)
                if distance_to_evaluated_point < min_distance:
                    min_distance = distance_to_evaluated_point
                if distance_to_evaluated_point < 1e-2/np.sqrt(X_evaluated.shape[1]):
                    use_suggested_sample = False
                i += 1
            
            print('Minimum distance to previously evaluated point is: {}'.format(min_distance))
            
        if not use_suggested_sample:
            print('Suggested point is to close to previously evaluated point; swithching to max variance sampling policy.')
                        
            def posterior_variance(x):
                self.model.set_hyperparameters(0)
                var = self.model.posterior_variance(x)
                for h in range(1, self.number_of_gp_hyps_samples):
                    self.model.set_hyperparameters(h)
                    var += self.model.posterior_variance(x)
                var = var[:, 0]
                return -var
            
            x0 = initial_design('random', self.decision_context_space, 1)
            #suggested_sample =  self.decision_context_space_optimizer.optimize(x0, posterior_variance)[0]
            suggested_sample =  apply_optimizer(self.decision_context_space_optimizer, x0, f=posterior_variance, context_manager=self.decision_context_space_context_manager, space=self.decision_context_space)[0]
            
            
        suggested_sample = np.atleast_2d(suggested_sample)
        return suggested_sample
        
            
            
            