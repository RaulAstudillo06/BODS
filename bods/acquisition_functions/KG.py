# Copyright (c) 2019, Raul Astudillo

import numpy as np
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.stats import norm


class KG(AcquisitionBase):
    """
    Knowledge gradient acquisition function function.

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details.
    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer, scenario_distribution, utility=None, expectation_utility=None, cost_withGradients=None):
        self.optimizer = optimizer
        self.scenario_distribution = scenario_distribution
        self.utility = utility
        self.expectation_utility = expectation_utility
        super(KG, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            self.cost_withGradients = constant_cost_withGradients
        self.n_gp_hyps_samples = min(3, self.model.number_of_hyps_samples())
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
        self.Z_samples = np.random.normal(size=25)
        self.acq_mean = 0.
        self.acq_std = 1.

    def _compute_acq(self, X):
        """
        Computes the acquisition function at X.

        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        X = np.atleast_2d(X)
        parallel = True
        if parallel and X.shape[0] > 1:
            acqX = self._compute_acq_parallel(X)
        else:
            acqX = self._compute_acq_sequential(X)
        if False:#X.shape[0]>1:
            # Computes adequate constants to normalize the acquisition function.
            self.acq_mean = np.mean(acqX)
            self.acq_std = np.std(acqX)
            print('acq mean and std changed')
        acqX = (acqX-self.acq_mean)/self.acq_std
        return acqX

    def _compute_acq_sequential(self, X):
        """
        Computes the acquisition function sequentially at X.

        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        acqX = np.zeros((X.shape[0], 1))
        for h in range(self.n_gp_hyps_samples):
            self.model.set_hyperparameters(h)
            aux_sigma_tilde = (self.model.posterior_variance(X))**(-0.5)
            for n in range(X.shape[0]):
                x = np.atleast_2d(X[n, :])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)
                self.model.partial_precomputation_for_variance_conditioned_on_next_point(x)
                for l in range(self.utility_support_cardinality):
                    for Z in self.Z_samples:
                        # Inner function of the KG acquisition function.
                        def inner_func(D):
                            D = np.atleast_2d(D)
                            n_d = D.shape[0]
                            func_val = np.zeros((D.shape[0], 1))
                            cross_product_grid = np.vstack([np.append(d,theta) for theta in self.scenario_support for d in D])
                            mean = self.model.posterior_mean(cross_product_grid)
                            var = self.model.posterior_variance_conditioned_on_next_point(cross_product_grid)
                            mean += self.model.posterior_covariance_between_points_partially_precomputed(
                                cross_product_grid, x)*(aux_sigma_tilde[n,0]*Z)
                            for w in range(self.scenario_support_cardinality):
                                func_val[:, 0] += self.scenario_prob_dist[w] * self.expectation_utility.eval_func(mean[w * n_d:(w + 1) * n_d, 0], var[w * n_d:(w + 1) * n_d, 0], self.utility_support[l])
                            return -func_val
                        # Inner function and its gradient of the KG acquisition function.
                        def inner_func_with_gradient(d):
                            d = np.atleast_2d(d)
                            func_val = np.zeros((1, 1))
                            func_gradient = np.zeros(d.shape)
                            cross_product_grid = np.vstack(
                                [np.append(d, theta) for theta in self.scenario_support])
                            mean = self.model.posterior_mean(cross_product_grid)
                            var = self.model.posterior_variance_conditioned_on_next_point(
                                cross_product_grid)
                            mean += self.model.posterior_covariance_between_points_partially_precomputed(
                                cross_product_grid, x) * (aux_sigma_tilde[n,0]*Z)
                            # Gradient
                            mean_gradient = self.model.posterior_mean_gradient(cross_product_grid)[:,:d.shape[1]]
                            var_gradient = self.model.posterior_variance_gradient_conditioned_on_next_point(cross_product_grid)[:,:d.shape[1]]
                            mean_gradient += self.model.posterior_covariance_gradient_partially_precomputed(cross_product_grid,x)[:,0,:d.shape[1]]*(aux_sigma_tilde[n,0]*Z)
                            for w in range(self.scenario_support_cardinality):
                                func_val[:, 0] += self.scenario_prob_dist[w]*self.expectation_utility.eval_func(mean[w,0], var[w,0], self.utility_support[l])
                                expectation_utility_gradient = self.expectation_utility.eval_gradient(mean[w,0], var[w,0], self.utility_support[l])
                                aux = np.vstack((mean_gradient[w, :], var_gradient[w, :]))
                                func_gradient += self.scenario_prob_dist[w]*np.matmul(expectation_utility_gradient, aux)
                            return -func_val, -func_gradient
                        acqX[n, 0] -= self.utility_prob_dist[l]*self.optimizer.optimize_inner_func(f=inner_func, f_df=inner_func_with_gradient)[1]
        acqX /= (self.n_gp_hyps_samples*len(self.Z_samples))
        return acqX

    def _compute_acq_parallel(self, X, n_cores=4):
        """
         Computes the acquisition function in parallel at X.

         :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
         :param n_cores: number of cores used. Default is 3.
        """
        n_x = len(X)
        acqX = np.zeros((n_x, 1))
        args = [[0 for i in range(2)] for j in range(n_x)]
        for n in range(n_x):
            args[n][0] = np.atleast_2d(X[n,:])
        pool = Pool(n_cores)
        for h in range(self.n_gp_hyps_samples):
            self.model.set_hyperparameters(h)
            aux_sigma_tilde = (self.model.posterior_variance(X))**(-0.5)
            for n in range(n_x):
                args[n][1] = aux_sigma_tilde[n, 0]
            acqX += np.atleast_2d(pool.map(self._compute_acq_parallel_helper, args))
        acqX /= (self.n_gp_hyps_samples*len(self.Z_samples))
        return acqX

    def _compute_acq_parallel_helper(self, args):
        """
         Helper function for computing the acquisition function in parallel.

         :param args: precomputed quantities required for computing the acquisition function. See _compute_acq_parallel.
        """
        x = args[0]
        aux_sigma_tilde = args[1]
        self.model.partial_precomputation_for_covariance(x)
        self.model.partial_precomputation_for_covariance_gradient(x)
        self.model.partial_precomputation_for_variance_conditioned_on_next_point(x)
        acqx = 0
        for l in range(self.utility_support_cardinality):
            for Z in self.Z_samples:
                # Inner function of the KG acquisition function.
                def inner_func(D):
                    D = np.atleast_2d(D)
                    n_d = D.shape[0]
                    func_val = np.zeros((D.shape[0], 1))
                    cross_product_grid = np.vstack([np.append(d,theta) for theta in self.scenario_support for d in D])
                    mean = self.model.posterior_mean(cross_product_grid)
                    var = self.model.posterior_variance_conditioned_on_next_point(cross_product_grid)
                    mean += self.model.posterior_covariance_between_points_partially_precomputed(
                        cross_product_grid, x)*(aux_sigma_tilde*Z)
                    for w in range(self.scenario_support_cardinality):
                        func_val[:, 0] += self.scenario_prob_dist[w] * self.expectation_utility.eval_func(mean[w * n_d:(w + 1) * n_d, 0], var[w * n_d:(w + 1) * n_d, 0], self.utility_support[l])
                    return -func_val
                # Inner function and its gradient of the KG acquisition function.
                def inner_func_with_gradient(d):
                    d = np.atleast_2d(d)
                    func_val = np.zeros((1, 1))
                    func_gradient = np.zeros(d.shape)
                    cross_product_grid = np.vstack(
                        [np.append(d, theta) for theta in self.scenario_support])
                    mean = self.model.posterior_mean(cross_product_grid)
                    var = self.model.posterior_variance_conditioned_on_next_point(
                        cross_product_grid)
                    mean += self.model.posterior_covariance_between_points_partially_precomputed(
                        cross_product_grid, x) * (aux_sigma_tilde*Z)
                    # Gradient
                    mean_gradient = self.model.posterior_mean_gradient(cross_product_grid)[:,:d.shape[1]]
                    var_gradient = self.model.posterior_variance_gradient_conditioned_on_next_point(cross_product_grid)[:,:d.shape[1]]
                    mean_gradient += self.model.posterior_covariance_gradient_partially_precomputed(cross_product_grid,x)[:,0,:d.shape[1]]*(aux_sigma_tilde*Z)
                    for w in range(self.scenario_support_cardinality):
                        func_val[:, 0] += self.scenario_prob_dist[w]*self.expectation_utility.eval_func(mean[w,0], var[w,0], self.utility_support[l])
                        expectation_utility_gradient = self.expectation_utility.eval_gradient(mean[w,0], var[w,0], self.utility_support[l])
                        aux = np.vstack((mean_gradient[w, :], var_gradient[w, :]))
                        func_gradient += self.scenario_prob_dist[w]*np.matmul(expectation_utility_gradient, aux)
                    return -func_val, -func_gradient
                acqx -= self.utility_prob_dist[l]*self.optimizer.optimize_inner_func(f=inner_func, f_df=inner_func_with_gradient)[1]

        return acqx[0]

    def _compute_acq_withGradients(self, X):
        """
        Computes the acquisition function and its gradient at X.

        :param X: point at which the acquisition function is evaluated. Should be a 2d array. WARNIG: note that this
        assumes X is a single point and not multiple as in the previous functions.
        """
        X = np.atleast_2d(X)
        acqX = np.zeros((X.shape[0], 1))
        dacq_dX  = np.zeros(X.shape)
        Z_samples = self.Z_samples #np.random.normal(size=5)
        for h in range(self.n_gp_hyps_samples):
            self.model.set_hyperparameters(h)
            inv_sqrt_varX = (self.model.posterior_variance(X)) ** (-0.5)
            inv_varX_noiseless = np.reciprocal(self.model.posterior_variance_noiseless(X))
            dvar_dX = self.model.posterior_variance_gradient(X)
            for n in range(X.shape[0]):
                x = np.atleast_2d(X[n, :])
                self.model.partial_precomputation_for_covariance(x)
                self.model.partial_precomputation_for_covariance_gradient(x)
                self.model.partial_precomputation_for_variance_conditioned_on_next_point(x)
                for l in range(self.utility_support_cardinality):
                    for Z in Z_samples:
                        aux_sigma_tilde = inv_sqrt_varX[n,0]*Z

                        # Inner function of the KG acquisition function.
                        def inner_func(D):
                            D = np.atleast_2d(D)
                            n_d = D.shape[0]
                            func_val = np.zeros((D.shape[0], 1))
                            cross_product_grid = np.vstack([np.append(d,theta) for theta in self.scenario_support for d in D])
                            mean = self.model.posterior_mean(cross_product_grid)
                            var = self.model.posterior_variance_conditioned_on_next_point(cross_product_grid)
                            mean += self.model.posterior_covariance_between_points_partially_precomputed(
                                cross_product_grid, x)*aux_sigma_tilde
                            for w in range(self.scenario_support_cardinality):
                                func_val[:, 0] += self.scenario_prob_dist[w] * self.expectation_utility.eval_func(mean[w * n_d:(w + 1) * n_d, 0], var[w * n_d:(w + 1) * n_d, 0], self.utility_support[l])
                            return -func_val
                        # Inner function and its gradient of the KG acquisition function.
                        def inner_func_with_gradient(d):
                            d = np.atleast_2d(d)
                            func_val = np.zeros((1, 1))
                            func_gradient = np.zeros(d.shape)
                            cross_product_grid = np.vstack(
                                [np.append(d, theta) for theta in self.scenario_support])
                            mean = self.model.posterior_mean(cross_product_grid)
                            var = self.model.posterior_variance_conditioned_on_next_point(
                                cross_product_grid)
                            mean += self.model.posterior_covariance_between_points_partially_precomputed(
                                cross_product_grid, x) * aux_sigma_tilde
                            # Gradient
                            mean_gradient = self.model.posterior_mean_gradient(cross_product_grid)[:,:d.shape[1]]
                            var_gradient = self.model.posterior_variance_gradient_conditioned_on_next_point(cross_product_grid)[:,:d.shape[1]]
                            mean_gradient += self.model.posterior_covariance_gradient_partially_precomputed(cross_product_grid,x)[:,0,:d.shape[1]]*aux_sigma_tilde
                            for w in range(self.scenario_support_cardinality):
                                func_val[:, 0] += self.scenario_prob_dist[w]*self.expectation_utility.eval_func(mean[w,0], var[w,0], self.utility_support[l])
                                expectation_utility_gradient = self.expectation_utility.eval_gradient(mean[w,0], var[w,0], self.utility_support[l])
                                aux = np.vstack((mean_gradient[w, :], var_gradient[w, :]))
                                func_gradient += self.scenario_prob_dist[w]*np.matmul(expectation_utility_gradient, aux)
                            return -func_val, -func_gradient
                        d_opt, opt_val = self.optimizer.optimize_inner_func(f=inner_func, f_df=inner_func_with_gradient)
                        acqX[n, 0] -= self.utility_prob_dist[l] * opt_val
                        #
                        cross_product_grid = np.vstack([np.append(d_opt, theta) for theta in self.scenario_support])
                        cross_cov = self.model.posterior_covariance_between_points_partially_precomputed(cross_product_grid,x)[:,0]
                        dcross_cov_dx = self.model.posterior_covariance_gradient(x,cross_product_grid)[0,:,:]
                        mean = self.model.posterior_mean(cross_product_grid)[:,0]
                        mean += cross_cov * aux_sigma_tilde
                        var = self.model.posterior_variance_conditioned_on_next_point(cross_product_grid)[:,0]
                        # mean gradient
                        tmp1 = (-0.5*Z*inv_sqrt_varX[n,:]**3)*cross_cov
                        mean_gradient = aux_sigma_tilde*dcross_cov_dx + np.tensordot(tmp1, dvar_dX[n,:], axes=0)
                        tmp2 = inv_varX_noiseless[n,0] * cross_cov
                        var_gradient = np.multiply(dcross_cov_dx.T,-2*tmp2).T + np.tensordot(np.square(tmp2), dvar_dX[n,:], axes=0)
                        marginal_acqu_grad = 0
                        for w in range(self.scenario_support_cardinality):
                            expectation_utility_gradient = self.expectation_utility.eval_gradient(mean[w], var[w], self.utility_support[l])
                            aux = np.vstack((mean_gradient[w, :], var_gradient[w, :]))
                            marginal_acqu_grad += self.scenario_prob_dist[w]*np.matmul(expectation_utility_gradient, aux)
                        dacq_dX[n, :] += self.utility_prob_dist[l] * marginal_acqu_grad
        acqX /= (self.n_gp_hyps_samples * len(Z_samples))
        dacq_dX /= (self.n_gp_hyps_samples * len(Z_samples))
        acqX = (acqX-self.acq_mean)/self.acq_std
        dacq_dX /= self.acq_std
        return acqX, dacq_dX

    def update_Z_samples(self):
        """
        Updates the standard Gaussian samples required for computing the KG acquisition function.
        """
        print('Z samples have changed')
        self.Z_samples = np.random.normal(size=len(self.Z_samples))
        #pass
