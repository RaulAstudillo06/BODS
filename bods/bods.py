import numpy as np
import csv
import time
import matplotlib.pyplot as plt
from aux_software.GPyOpt.util.duplicate_manager import DuplicateManager
from aux_software.GPyOpt.core.errors import InvalidConfigError
from aux_software.GPyOpt.core.task.cost import CostModel
from aux_software.GPyOpt.optimization.acquisition_optimizer import ContextManager
from aux_software.GPyOpt.optimization import GeneralOptimizer
#from aux_software.plotting_services import plot_convergence, plot_acquisition, integrated_plot
from pathos.multiprocessing import ProcessingPool as Pool
from copy import deepcopy
from scipy.stats import norm


class BODS(object):
    """
    Runner of the Bayesian-optimization-for-decision-support" loop. This class wraps the optimization loop around the different handlers.
    :param model: GPyOpt model class.
    :param space: GPyOpt space class.
    :param objective: GPyOpt objective class.
    :param acquisition: GPyOpt acquisition class.
    :param evaluator: GPyOpt evaluator class.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    :param cost: GPyOpt cost class (default, none).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, False).
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    :param utility: utility function (given by a and l). See utility.py for more information.
    :param scenario_distribution: distribution over $\Theta$. See scenario_distribution.py for more information.
    """

    def __init__(self, model, optimization_space, objective, sampling_policy, acquisition, evaluator, utility, scenario_distribution, X_init, Y_init=None, cost=None,
                 normalize_Y=False, model_update_interval=1, expectation_utility=None):
        self.model = model
        self.optimization_space = optimization_space
        self.decision_context_space = optimization_space.decision_context_space
        self.decision_space = optimization_space.decision_space
        self.objective = objective
        self.sampling_policy = sampling_policy
        self.acquisition = acquisition
        self.evaluator = evaluator
        self.utility = utility
        self.scenario_distribution = scenario_distribution
        self.X = X_init
        self.Y = Y_init
        self.cost = CostModel(cost)
        self.normalize_Y = normalize_Y
        self.model_update_interval = model_update_interval
        self.expectation_utility = expectation_utility
        #
        self.historical_optimal_values = []
        self.number_of_gp_hyps_samples = min(10, self.model.number_of_hyps_samples())
        self.decision_space_dim = self.optimization_space.decision_space_dim()
        self.utility_prob_dist = self.utility.parameter_distribution
        self.full_scenario_support = True
        self.scenario_support =  scenario_distribution.support
        self.scenario_prob_dist = scenario_distribution.prob_dist
        if self.full_scenario_support:
            self.scenario_support_cardinality = len(self.scenario_support)
        self.utility_support = utility.parameter_distribution.support
        self.utility_prob_dist = utility.parameter_distribution.prob_dist
        self.full_utility_support = self.utility.parameter_distribution.use_full_support
        if self.full_utility_support:
            self.utility_support_cardinality = len(self.utility_support)
            self.current_marginal_best_point = [0]*self.utility_support_cardinality
            self.historical_marginal_best_points = []
            self.val_of_historical_marginal_best_points = []
            if self.full_scenario_support:
                self.f_at_historical_marginal_optima = []
        self.evaluation_optimizer = GeneralOptimizer(optimizer='lbfgs', space=self.decision_space, parallel=False)

    def _current_max_value(self):
        """
        Computes real underlying value at current optimum.
        """
        val = 0
        scenario_marginal_optimal_val = []
        if self.full_utility_support:
            marginal_argmaxs = []
            val_at_marginal_argmaxs = []
            for l in range(self.utility_support_cardinality):
                print('Utility parameter: {}'.format(self.utility_support[l]))
                marginal_val = 0
                argmax = self._current_marginal_argmax(self.utility_support[l])
                self.current_marginal_best_point[l] = argmax
                print('Current marginal optimum: {}'.format(argmax))
                marginal_argmaxs.append(argmax)
                print('Values of f at current marginal optimum for all scenarios:')
                for w in range(self.scenario_support_cardinality):
                    argmax_context = np.atleast_2d(np.append(argmax[0,:], self.scenario_support[w]))
                    objective_val = np.asscalar(self.objective.evaluate(argmax_context)[0])
                    scenario_marginal_optimal_val.append(objective_val)
                    print('Marginal objective value for scenario {}: {}'.format(self.scenario_support[w], objective_val))
                    marginal_val += self.scenario_prob_dist[w]*(self.utility.eval_func(objective_val, self.utility_support[l]))
                print('Current marginal optimal value: {}'.format(marginal_val))    
                val_at_marginal_argmaxs.append(marginal_val)
                val += self.utility_prob_dist[l]*marginal_val
            self.historical_marginal_best_points.append(marginal_argmaxs)
            self.val_of_historical_marginal_best_points.append(val_at_marginal_argmaxs)
        print('Current optimal value: {}'.format(val))
        self.historical_optimal_values.append(val)
        self.f_at_historical_marginal_optima.append(scenario_marginal_optimal_val)


    def _current_marginal_argmax(self, utility_parameter):
        """
        Computes argmaxE_nE_{theta}[U_j(v(d,theta))].

        :param utility_sample_id: id to identify the utility function (that is, j).
        """
        def val_func(D):
            D = np.atleast_2d(D)
            n_d = D.shape[0]
            func_val = np.zeros((D.shape[0], 1))
            cross_product_grid = np.vstack([np.append(d, theta) for theta in self.scenario_support for d in D])
            for h in range(self.number_of_gp_hyps_samples):
                self.model.set_hyperparameters(h)
                mean, var = self.model.predict_noiseless(cross_product_grid)
                expected_utility = self.expectation_utility.eval_func(mean, var, utility_parameter) 
                for w in range(self.scenario_support_cardinality):
                    func_val[:, 0] += self.scenario_prob_dist[w] * (expected_utility[w*n_d:(w + 1)*n_d, 0])
            func_val /= self.number_of_gp_hyps_samples
            return -func_val

        def val_func_with_gradient(d):
            d = np.atleast_2d(d)
            func_val = np.zeros((1, 1))
            func_gradient = np.zeros(d.shape)
            cross_product_grid = np.vstack(
                [np.append(d, theta) for theta in self.scenario_support])
            for h in range(self.number_of_gp_hyps_samples):
                self.model.set_hyperparameters(h)
                mean, var = self.model.predict_noiseless(cross_product_grid)
                mean_gradient = self.model.posterior_mean_gradient(cross_product_grid)[:, :d.shape[1]]
                var_gradient = self.model.posterior_variance_gradient(cross_product_grid)[:, :d.shape[1]]
                for w in range(self.scenario_support_cardinality):
                    expectation_utility = self.expectation_utility.eval_func(mean[w,0], var[w,0], utility_parameter)
                    func_val += self.scenario_prob_dist[w] * (expectation_utility)
                    expectation_utility_gradient = self.expectation_utility.eval_gradient(mean[w,0], var[w,0], utility_parameter)
                    aux = np.vstack((mean_gradient[w, :], var_gradient[w, :]))
                    #print(expectation_utility_gradient)
                    #print(aux)
                    func_gradient += self.scenario_prob_dist[w]*np.matmul(expectation_utility_gradient, aux)
            func_val /= self.number_of_gp_hyps_samples
            func_gradient /= self.number_of_gp_hyps_samples
            return -func_val, -func_gradient

        argmax = self.evaluation_optimizer.optimize(f=val_func, f_df=val_func_with_gradient, parallel=False)[0]
        return argmax

    def run_optimization(self, max_iter=1, parallel=False, plot=False, filename=None, max_time=np.inf,
                         context=None, verbosity=False):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param filename: filename of the file the optimization results are saved (default, None).
        """

        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.verbosity = verbosity
        self.filename = filename
        self.context = context

        # --- Setting up stop conditions
        if (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # --- Initial function evaluation
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)
        # --- Initialize model
        self.model.updateModel(self.X, self.Y)
        self.model.get_model_parameters_names()
        self.model.get_model_parameters()

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.num_acquisitions = 0
        self.cum_time = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y

        # --- Initialize time cost of the evaluations
        while (self.max_time > self.cum_time) and (self.num_acquisitions < self.max_iter):
            self.suggested_sample = self._compute_next_evaluations()
            try:
                self.acquisition.update_Z_samples()
            except:
                pass
            # --- Augment XS
            self.X = np.vstack((self.X, self.suggested_sample))

            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            print('Experiment: ' + filename[0])
            print('Sampling policy: ' + filename[1])
            print('Replication id: ' + filename[2])
            print('Acquisition number: {}'.format(self.num_acquisitions + 1))
            self.evaluate_objective()
            if filename is not None:
                self.save_evaluations(filename)
            # --- Update model
            if (self.num_acquisitions % self.model_update_interval) == 0:
                self._update_model()
            self.model.get_model_parameters_names()
            self.model.get_model_parameters()
            self._current_max_value()
            if filename is not None:
                self.save_results(filename)
            self._save_baseline_points
            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1

        self.f_at_historical_marginal_optima = np.asarray(self.f_at_historical_marginal_optima)
        self.historical_marginal_best_points = np.asarray(self.historical_marginal_best_points)
        self.val_of_historical_marginal_best_points = np.asarray(self.val_of_historical_marginal_best_points)

    def evaluate_objective(self):
        """
        Evaluates the objective
        """
        print('Suggested point to evaluate: {}'.format(self.suggested_sample))
        self.Y_new, cost_new = self.objective.evaluate(self.suggested_sample)
        self.cost.update_cost_model(self.suggested_sample, cost_new)
        self.Y = np.vstack((self.Y, self.Y_new))

    def _compute_next_evaluations(self, pending_zipped_X=None, ignored_zipped_X=None):
        """
        Computes the location of the new evaluation (optimizes the acquisition in the standard case).
        :param pending_zipped_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet).
        :param ignored_zipped_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again.
        :return:
        """
        if self.acquisition is not None:
            ## --- Update the context if any
            self.acquisition.optimizer.context_manager = ContextManager(self.decision_context_space, self.context)
    
            ### We zip the value in case there are categorical variables
            suggested_sample = self.decision_context_space.zip_inputs(self.evaluator.compute_batch(duplicate_manager=None))
        else:
            suggested_sample = self.sampling_policy.suggest_sample()
        return suggested_sample

    def _update_model(self):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """

        ### --- input that goes into the model (is unziped in case there are categorical variables)
        X_inmodel = self.decision_context_space.unzip_inputs(self.X)
        Y_inmodel = np.copy(self.Y)
        self.model.updateModel(X_inmodel, Y_inmodel)
        
    def _save_baseline_points(self):
        """
        """
        if self.acquisition is not None:
            if self.full_utility_support:
                if self.full_scenario_support:
                    baseline_points = []
                    for l in range(self.utility_support_cardinality):
                        for w in range(self.scenario_support_cardinality):
                            baseline_points.append(np.append(self.current_marginal_best_point[l], self.scenario_support[w]))
                    self.acquisition.optimizer.baseline_points = baseline_points

    def get_evaluations(self):
        return self.X.copy(), self.Y.copy()

    def save_evaluations(self, filename):
        """
        """
        experiment_folder_name = './results/' + filename[0]
        experiment_name = filename[0] + '_' + filename[1] + '_' + filename[2]
        np.savetxt(experiment_folder_name + '/X/' + experiment_name + '_X.txt', self.X)
        np.savetxt(experiment_folder_name + '/Y/' + experiment_name + '_Y.txt', self.Y)

    def save_results(self, filename):
        """
        """
        experiment_folder_name = './results/' + filename[0]
        experiment_name = filename[0] + '_' + filename[1] + '_' + filename[2]
        aux_filename = experiment_folder_name + '/' + experiment_name + '.txt'
        results = np.atleast_1d(self.historical_optimal_values)
        np.savetxt(aux_filename, results)
        #aux_filename = experiment_folder_name + '/historical marginal optima/' + experiment_name + '_historical_marginal_optima.txt'
        #np.savetxt(aux_filename, self.historical_marginal_best_points)
        aux_filename = experiment_folder_name + '/values at historical marginal optima/' + experiment_name + '_values_at_historical_marginal_optima.txt'
        np.savetxt(aux_filename, self.val_of_historical_marginal_best_points)
        aux_filename = experiment_folder_name + '/f at historical marginal optima/' + experiment_name + '_f_at_historical_marginal_optima.txt'
        np.savetxt(aux_filename, self.f_at_historical_marginal_optima)
