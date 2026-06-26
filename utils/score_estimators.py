import torch
import abc
import samplers.mala as mala
from utils.densities import Distribution
import utils.optimizers as optimizers
import samplers.rejection_sampler as rejection_sampler
import samplers.ula as ula
import numpy as np


class ScoreEstimator(abc.ABC):
    def __init__(self, dist: Distribution,
                 sde, device, def_num_batches=1,
                 def_num_samples=10000) -> None:
        self.sde = sde
        self.dist = dist
        self.device = device
        self.default_num_batches = def_num_batches
        self.default_num_samples = def_num_samples
        self.dim = self.dist.dim

    @abc.abstractmethod
    def score_estimator(self, x, tt, num_batches=None, num_rej_samples=None):
        pass


class ADDDS(ScoreEstimator):

    def __init__(self, dist: Distribution, sde, device,
                 config,
                 def_num_batches=1,
                 def_num_rej_samples=10000,
                 max_iters_opt=50,
                 optimizer_reg_lambda=0.01
                 ) -> None:
        super().__init__(dist, sde, device, def_num_batches, def_num_rej_samples)
        self.max_iters_opt = max_iters_opt
        self.optimizer_reg_lambda = optimizer_reg_lambda
        self.config = config
        self.vals_history = []
        self.grads_history = []
        self.call_count = 0
        self.avg_acceptance_rate = 0.0
        self.std_acceptance_rate = 0.0
        self.acceptance_rates_history = []


    def score_estimator(self, x, tt, num_batches=None, num_samples=None, reg_val_scalar_for_model=None, **kwargs):
        self.call_count += 1
        self.dist.keep_minimizer = True

        initial_guess_for_minimizer = torch.randn(self.dist.dim, device=self.device)

        current_call_vals = []
        current_call_grads = []
        current_step_rates = []

        potential = lambda x_opt: -self.dist.log_prob(x_opt)
        grad_potential = lambda x_opt: -self.dist.grad_log_prob(x_opt)



        current_minimizer = optimizers.newton_conjugate_gradient_ADDDS(
            initial_guess_for_minimizer,
            potential,
            max_iters=self.max_iters_opt,
            regularization_objective_value=reg_val_scalar_for_model,
            regularization_lambda=self.optimizer_reg_lambda,
            grad_potential_func=grad_potential,
            vals_list=current_call_vals,
            grads_list=current_call_grads
        )



        if current_call_vals:
            self.vals_history.append(np.mean(current_call_vals))
        if current_call_grads:
            self.grads_history.append(np.mean(current_call_grads))

        if self.call_count >= (self.config.disc_steps - 1):
            if self.vals_history and self.grads_history:
                val_mean = np.mean(self.vals_history)
                val_std = np.std(self.vals_history)
                grad_mean = np.mean(self.grads_history)
                grad_std = np.std(self.grads_history)

               

            self.vals_history = []
            self.grads_history = []
            self.call_count = 0

        sde_T = self.sde.T()
        sde_delta = self.sde.delta
        
        self.dist.log_prob(current_minimizer)

        self.dist.potential_minimizer = current_minimizer

        scaling = self.sde.scaling(tt)
        variance_conv = (1 / scaling) ** 2 - 1
        score_estimate = torch.zeros_like(x)
        num_batches = self.default_num_batches if num_batches is None else num_batches
        num_samples = self.default_num_samples if num_samples is None else num_samples

        assert num_batches > 0 and num_samples > 0, 'Number of samples needs to be a positive integer'

        mean_estimate = 0
        num_good_samples = torch.zeros((x.shape[0], 1), device=self.device)
        for _ in range(num_batches):

            samples_from_p0t, acc_idx, batch_acc_rate = rejection_sampler.get_samples(
                x / scaling, variance_conv, self.dist, num_samples, self.device
            )
            current_step_rates.append(batch_acc_rate)

            num_good_samples += torch.sum(acc_idx, dim=(1, 2)).unsqueeze(-1).to(torch.double) / self.dim
            mean_estimate += torch.sum(samples_from_p0t * acc_idx, dim=1)

        if len(current_step_rates) > 0:
            step_avg = sum(current_step_rates) / len(current_step_rates)
            self.acceptance_rates_history.append(step_avg)
            self.avg_acceptance_rate = np.mean(self.acceptance_rates_history)
            if len(self.acceptance_rates_history) > 1:
                self.std_acceptance_rate = np.std(self.acceptance_rates_history)
            else:
                self.std_acceptance_rate = 0.0

        num_good_samples[num_good_samples == 0] += 1
        mean_estimate /= num_good_samples
        score_estimate = (scaling * mean_estimate - x) / (1 - scaling ** 2 + 1e-9)
        return score_estimate



class ADDDS_Hessian(ScoreEstimator):
    def __init__(self, dist: Distribution, sde, device,
                 config, def_num_batches=1, def_num_rej_samples=10000,
                 max_iters_opt=50, optimizer_reg_lambda=0.01) -> None:
        super().__init__(dist, sde, device, def_num_batches, def_num_rej_samples)
        self.max_iters_opt = max_iters_opt
        self.optimizer_reg_lambda = optimizer_reg_lambda
        self.config = config

    def score_estimator(self, x, tt, num_batches=None, num_samples=None, **kwargs):
        self.dist.keep_minimizer = True
        initial_guess_for_minimizer = torch.randn(self.dist.dim, device=self.device)
        potential = lambda x_opt: -self.dist.log_prob(x_opt)

        current_minimizer = optimizers.newton_conjugate_gradient_Hessian(
            initial_guess_for_minimizer,
            potential,
            max_iters=self.max_iters_opt,
            regularization_lambda=self.optimizer_reg_lambda
        )
        self.dist.log_prob(current_minimizer)
        scaling = self.sde.scaling(tt)
        variance_conv = (1 / scaling) ** 2 - 1
        num_batches = self.default_num_batches if num_batches is None else num_batches
        num_samples = self.default_num_samples if num_samples is None else num_samples

        mean_estimate = 0
        num_good_samples = torch.zeros((x.shape[0], 1), device=self.device)
        for _ in range(num_batches):
            samples_from_p0t, acc_idx, batch_acc_rate = rejection_sampler.get_samples(x / scaling, variance_conv,
                                                                      self.dist, num_samples, self.device)
            num_good_samples += torch.sum(acc_idx, dim=(1, 2)).unsqueeze(-1).to(torch.double) / self.dim
            mean_estimate += torch.sum(samples_from_p0t * acc_idx, dim=1)
        num_good_samples[num_good_samples == 0] += 1
        mean_estimate /= num_good_samples
        score_estimate = (scaling * mean_estimate - x) / (1 - scaling ** 2 + 1e-9)
        return score_estimate



class ZODMC_ScoreEstimator(ScoreEstimator):
    def __init__(self, dist: Distribution, sde, device,
                 def_num_batches=1,
                 def_num_rej_samples=10000,
                 max_iters_opt=50
                 ) -> None:
        super().__init__(dist, sde, device, def_num_batches, def_num_rej_samples)
        dist.keep_minimizer = True
        self.avg_acceptance_rate = 0.0
        self.std_acceptance_rate = 0.0
        self.acceptance_rates_history = []
        minimizer = optimizers.newton_conjugate_gradient(torch.randn(dist.dim, device=device),
                                                         lambda x: -self.dist.log_prob(x),
                                                         grad_potential_func=lambda x: -self.dist.grad_log_prob(x),
                                                         max_iters=max_iters_opt)
        potential_value_at_minimum = -self.dist.log_prob(minimizer)
        grad_potential_at_minimum = -self.dist.grad_log_prob(minimizer)
        l2_norm_of_grad = torch.linalg.norm(grad_potential_at_minimum)
        dist.log_prob(minimizer)

    def score_estimator(self, x, tt, num_batches=None, num_samples=None):
            scaling = self.sde.scaling(tt)
            variance_conv = (1 / scaling) ** 2 - 1
            num_batches = self.default_num_batches if num_batches is None else num_batches
            num_samples = self.default_num_samples if num_samples is None else num_samples
            assert num_batches > 0 and num_samples > 0, 'Number of samples needs to be a positive integer'
            current_step_rates = []
            mean_estimate = 0
            num_good_samples = torch.zeros((x.shape[0], 1), device=self.device)
            for _ in range(num_batches):
                samples_from_p0t, acc_idx, batch_acc_rate = rejection_sampler.get_samples(
                    x / scaling, variance_conv, self.dist, num_samples, self.device
                )
                current_step_rates.append(batch_acc_rate)
                num_good_samples += torch.sum(acc_idx, dim=(1, 2)).unsqueeze(-1).to(torch.double) / self.dim
                mean_estimate += torch.sum(samples_from_p0t * acc_idx, dim=1)

            if len(current_step_rates) > 0:
                step_avg = sum(current_step_rates) / len(current_step_rates)
                self.acceptance_rates_history.append(step_avg)
                self.avg_acceptance_rate = np.mean(self.acceptance_rates_history)
                if len(self.acceptance_rates_history) > 1:
                    self.std_acceptance_rate = np.std(self.acceptance_rates_history)
                else:
                    self.std_acceptance_rate = 0.0
            num_good_samples[num_good_samples == 0] += 1
            mean_estimate /= num_good_samples
            score_estimate = (scaling * mean_estimate - x) / (1 - scaling ** 2 + 1e-9)
            return score_estimate


class RDMC_ScoreEstimator(ScoreEstimator):
    def __init__(self, dist: Distribution, sde, device,
                 def_num_batches=1,
                 def_num_samples=10000,
                 ula_step_size=0.01,
                 ula_steps=10,
                 initial_cond_normal=True) -> None:
        super().__init__(dist, sde, device, def_num_batches, def_num_samples)
        self.ula_step_size = ula_step_size
        self.ula_steps = ula_steps
        self.initial_cond_normal = initial_cond_normal

    def score_estimator(self, x, tt, **kwargs):
        scaling = self.sde.scaling(tt)
        inv_scaling = 1 / scaling
        variance_conv = inv_scaling ** 2 - 1
        num_samples = self.default_num_samples
        big_x = x.repeat_interleave(num_samples, dim=0)

        def grad_log_prob_0t(x0):
            return self.dist.grad_log_prob(x0) + scaling * (big_x - scaling * x0) / (1 - scaling ** 2 + 1e-9)

        mean_estimate = 0
        x0 = big_x
        for _ in range(self.default_num_batches):
            if self.initial_cond_normal:
                x0 = inv_scaling * big_x + torch.randn_like(big_x) * variance_conv ** .5
            samples_from_p0t = ula.get_ula_samples(x0, grad_log_prob_0t, self.ula_step_size, self.ula_steps)
            samples_from_p0t = samples_from_p0t.view((-1, num_samples, self.dim))
            mean_estimate += torch.sum(samples_from_p0t, dim=1)
        mean_estimate /= (self.default_num_batches * self.default_num_samples)

        score_estimate = (scaling * mean_estimate - x) / (1 - scaling ** 2 + 1e-9)
        return score_estimate


class RSDMC_ScoreEstimator(ScoreEstimator):
    def __init__(self, dist: Distribution, sde, device,
                 def_num_batches=1,
                 def_num_samples=10000,
                 ula_step_size=0.01,
                 ula_steps=10,
                 num_recursive_steps=3,
                 initial_cond_normal=True) -> None:
        super().__init__(dist, sde, device, def_num_batches, def_num_samples)
        self.ula_step_size = ula_step_size
        self.ula_steps = ula_steps
        self.initial_cond_normal = initial_cond_normal
        self.num_recursive_steps = num_recursive_steps

    def _recursive_langevin(self, x, tt, k=None):
        if k is None:
            k = self.num_recursive_steps
        if k == 0 or tt < .2:
            return self.dist.grad_log_prob(x)

        num_samples = self.default_num_samples
        scaling = self.sde.scaling(tt)
        h = self.ula_step_size
        big_x = x.repeat_interleave(num_samples, dim=0)
        x0 = big_x.detach().clone()
        for _ in range(self.ula_steps):
            score = self._recursive_langevin(x0, (k - 1) * tt / k, k - 1) + scaling * (big_x - scaling * x0) / (
                        1 - scaling ** 2 + 1e-9)
            x0 = x0 + h * score + (2 * h) ** .5 * torch.randn_like(x0)
        x0 = x0.view((-1, num_samples, self.dim))
        mean_estimate = x0.mean(dim=1)

        score_estimate = (scaling * mean_estimate - x) / (1 - scaling ** 2 + 1e-9)
        return score_estimate

    def score_estimator(self, x, tt, **kwargs):
        score_estimate = 0
        for _ in range(self.default_num_batches):
            score_estimate += self._recursive_langevin(x, tt, self.num_recursive_steps)
        score_estimate /= self.default_num_batches
        return score_estimate

class SLIPS_ScoreEstimator(ScoreEstimator):
    def __init__(self, dist: Distribution, sde, device,
                 def_num_batches=1,
                 def_num_samples=1000,
                 mala_step_size=0.01,
                 mala_steps=50,
                 initial_cond_normal=True) -> None:
        super().__init__(dist, sde, device, def_num_batches, def_num_samples)
        self.mala_step_size = mala_step_size
        self.mala_steps = mala_steps
        self.initial_cond_normal = initial_cond_normal

    def score_estimator(self, x, tt, num_batches=None, num_samples=None, **kwargs):
        scaling = self.sde.scaling(tt)
        inv_scaling = 1.0 / scaling
        variance_posterior_mala = (1.0 / scaling ** 2) - 1.0
        num_batches_eff = self.default_num_batches if num_batches is None else num_batches
        num_samples_eff = self.default_num_samples if num_samples is None else num_samples
        assert num_batches_eff > 0 and num_samples_eff > 0, 'Number of samples needs to be a positive integer'
        score_estimate_sum = torch.zeros_like(x)
        big_x = x.repeat_interleave(num_samples_eff, dim=0)
        mean_posterior_param = x / scaling
        var_posterior_param = (1 - scaling ** 2) / scaling ** 2
        if variance_posterior_mala <= 1e-6:

            pass
        big_x_cond = big_x / scaling

        def log_prob_posterior_fn(x0_batch):
            log_p0 = self.dist.log_prob(x0_batch)
            diff_sq = torch.sum((x0_batch - big_x_cond) ** 2, dim=-1, keepdim=True)
            return log_p0 - 0.5 * diff_sq / variance_posterior_mala

        def grad_log_prob_posterior_fn(x0_batch):
            grad_log_p0 = self.dist.grad_log_prob(x0_batch)
            return grad_log_p0 - (x0_batch - big_x_cond) / variance_posterior_mala

        for i_batch in range(num_batches_eff):
            if self.initial_cond_normal:
                x0_initial_mala = inv_scaling * big_x + torch.randn_like(big_x) * (
                            inv_scaling ** 2 - 1) ** .5 if inv_scaling > 1 else inv_scaling * big_x + torch.randn_like(
                    big_x)
            else:
                x0_initial_mala = big_x.detach().clone()
 
            samples_from_posterior = mala.get_mala_samples(
                x0_initial_mala,
                log_prob_posterior_fn,
                grad_log_prob_posterior_fn,
                self.mala_step_size,
                self.mala_steps,
                display_pbar=False
            )

            samples_from_posterior_reshaped = samples_from_posterior.view(x.shape[0], num_samples_eff, self.dim)
            score_estimate_sum += torch.sum(samples_from_posterior_reshaped, dim=1)

        mean_estimate_final = score_estimate_sum / (num_batches_eff * num_samples_eff)
        score = (scaling * mean_estimate_final - x) / (1 - scaling ** 2)
        return score


def get_score_function(config, dist : Distribution, sde, device):

    grad_logdensity = dist.grad_log_prob
    dim = dist.dim

    def get_recursive_langevin(x,tt,k=config.num_recursive_steps):
        if k == 0 or tt < .2:
            return grad_logdensity(x)

        num_samples = config.num_estimator_samples
        scaling = sde.scaling(tt)

        h = config.ula_step_size

        big_x = x.repeat_interleave(num_samples,dim=0)
        x0 = big_x.detach().clone()

        for _ in range(config.num_sampler_iterations):
            score = get_recursive_langevin(x0, (k-1) * tt/k,k-1) + scaling * (big_x - scaling * x0)/(1-scaling**2)
            x0 = x0 + h * score + (2*h)**.5 * torch.randn_like(x0)
        x0 = x0.view((-1,num_samples,dim))
        mean_estimate = x0.mean(dim=1)
        score_estimate = (scaling * mean_estimate - x)/(1 - scaling**2)
        return score_estimate


    if config.score_method == 'p0t' and config.p0t_method == 'rejection' and config.optimizer_reg == 'True':
        return ADDDS(dist,sde,device,
                     config,
                     def_num_batches=config.num_estimator_batches,
                     def_num_rej_samples=config.num_estimator_samples,
                     max_iters_opt = config.max_iters_optimization,
                     optimizer_reg_lambda = config.optimizer_reg_lambda).score_estimator

    elif config.score_method == 'p0t' and config.p0t_method == 'rejection_hessian':
        return ADDDS_Hessian(dist,sde,device,
                     config,
                     def_num_batches=config.num_estimator_batches,
                     def_num_rej_samples=config.num_estimator_samples,
                     max_iters_opt = config.max_iters_optimization,
                     optimizer_reg_lambda = config.optimizer_reg_lambda).score_estimator

    elif config.score_method == 'p0t' and config.p0t_method == 'rejection' and config.optimizer_reg == 'False':
        return ZODMC_ScoreEstimator(dist,sde,device,
                                    def_num_batches=config.num_estimator_batches,
                                    def_num_rej_samples=config.num_estimator_samples).score_estimator
    elif config.score_method == 'p0t' and config.p0t_method == 'ula':
        initial_cond_normal= True if config.rdmc_initial_condition.lower() == 'normal' else False
        return RDMC_ScoreEstimator(dist,sde,device,
                                def_num_batches=config.num_estimator_batches,
                                def_num_samples=config.num_estimator_samples,
                                ula_step_size=config.ula_step_size,
                                ula_steps=config.num_sampler_iterations,
                                initial_cond_normal=initial_cond_normal).score_estimator
    elif config.score_method == 'p0t' and config.p0t_method == 'mala':

        mala_step_size = getattr(config, 'slips_mala_step_size', 0.01)
        mala_steps = getattr(config, 'slips_mala_steps', 50)
        slips_initial_cond_normal = getattr(config, 'slips_initial_cond_normal', True)

        return SLIPS_ScoreEstimator(dist, sde, device,
                                    def_num_batches=config.num_estimator_batches,
                                    def_num_samples=config.num_estimator_samples,

                                    mala_step_size=mala_step_size,
                                    mala_steps=mala_steps,
                                    initial_cond_normal=slips_initial_cond_normal).score_estimator
    elif config.score_method == 'recursive':
        return RSDMC_ScoreEstimator(dist,sde,device,
                                def_num_batches=config.num_estimator_batches,
                                def_num_samples=config.num_estimator_samples,
                                ula_step_size=config.ula_step_size,
                                num_recursive_steps=config.num_recursive_steps,
                                ula_steps=config.num_sampler_iterations).score_estimator