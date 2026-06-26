import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp as scipy_logsumexp


def _minfunc_smc(log_dp, llh_values):
    lw = llh_values * np.exp(log_dp)
    lw_mean = scipy_logsumexp(lw) - np.log(len(lw))
    var = np.std(np.exp(lw - lw_mean))
    return (var - 1) ** 2.


def smc_sampler_for_zodmc(
        target_dist,
        dim,
        n_particles,
        n_mcmc_steps_per_particle,
        beta_smc,
        device,
        oracle_complexity_info=None,
        noise_std_per_epoch=0.1,
        config=None,
        score_model=None
):
    particles = torch.randn((n_particles, dim), device=device) * 5.0
    p_anneal = 0.0
    dp_anneal = 0.1
    epoch = 1

    while p_anneal < 1.0:
        particles_np = particles.cpu().numpy()
        log_likelihood_values = []
        for i in range(n_particles):
            sample_torch = particles[i:i + 1].to(device)
            try:
                log_prob_val = target_dist.log_prob(sample_torch).sum()
                log_likelihood_values.append(log_prob_val.item())
            except Exception as e:
                log_likelihood_values.append(-np.inf)

        llh_np = np.array(log_likelihood_values)
        llh_np[np.isneginf(llh_np)] = -1e9


        if len(llh_np[np.isfinite(llh_np)]) > 1:
            log_dp_initial = np.log(dp_anneal)
            res = minimize(lambda x: _minfunc_smc(x, llh_np), log_dp_initial, method='COBYLA',
                           options={'maxiter': 100, 'rhobeg': 0.1})
            dp_anneal = np.exp(res.x).item()
        else:
            dp_anneal = 0.01

        if p_anneal + dp_anneal > 1.0:
            dp_anneal = 1.0 - p_anneal
        p_anneal += dp_anneal

        weights = np.exp(llh_np * dp_anneal)
        weights[np.isnan(weights)] = 0
        weights[np.isinf(weights)] = 0

        sum_weights = np.sum(weights)
        if sum_weights == 0 or not np.isfinite(sum_weights):
            normalized_weights = np.ones(n_particles) / n_particles
        else:
            normalized_weights = weights / sum_weights
        try:
            indices = np.random.choice(np.arange(n_particles), size=n_particles, p=normalized_weights)
        except ValueError as e:
            indices = np.random.choice(np.arange(n_particles), size=n_particles)
        particles = particles[indices]
        if particles.shape[0] > 1:
            if particles.ndim == 2 and particles.shape[1] > 0:
                if dim == 1:
                    cov_matrix_torch = torch.var(particles, dim=0, unbiased=False).unsqueeze(0) * torch.eye(dim, device=device)
                    if cov_matrix_torch.nelement() == 0: cov_matrix_torch = torch.eye(dim, device=device) * 1e-6
                else:
                    cov_matrix_torch = torch.cov(particles.T)

                cov_matrix_torch += torch.eye(dim, device=device) * 1e-6
            else:
                cov_matrix_torch = torch.eye(dim, device=device)
        else:
            cov_matrix_torch = torch.eye(dim, device=device)

        try:
            L = torch.linalg.cholesky((beta_smc ** 2) * cov_matrix_torch)
        except RuntimeError as e:
            std_devs = torch.std(particles, dim=0)
            std_devs[std_devs < 1e-3] = 1e-3
            cov_matrix_diag = torch.diag((beta_smc ** 2) * std_devs ** 2)
            L = torch.linalg.cholesky(cov_matrix_diag + torch.eye(dim, device=device) * 1e-6)


        accepted_count_total = 0
        for _ in range(n_mcmc_steps_per_particle):
            for j in range(n_particles):
                current_particle = particles[j].clone()
                d_proposal = (L @ torch.randn(dim, device=device))
                candidate_particle = current_particle + d_proposal
                try:
                    log_prob_current = target_dist.log_prob(current_particle.unsqueeze(0)).sum()
                    log_prob_candidate = target_dist.log_prob(candidate_particle.unsqueeze(0)).sum()
                    if not torch.isfinite(log_prob_current): log_prob_current = -torch.inf
                    if not torch.isfinite(log_prob_candidate): log_prob_candidate = -torch.inf
                    log_alpha = log_prob_candidate - log_prob_current
                    if torch.log(torch.rand(1, device=device)) < log_alpha:
                        particles[j] = candidate_particle
                        accepted_count_total += 1
                except Exception:
                    pass

        acceptance_rate = accepted_count_total / (n_particles * n_mcmc_steps_per_particle)
        if config is not None and score_model is not None:
            perturbation_mode = getattr(config, 'smc_perturbation_mode', 'none').lower()
            sde_T = config.T
            current_t = sde_T * (1 - p_anneal)
            if 'pgd' in perturbation_mode:
                pgd_eps = getattr(config, 'smc_pgd_eps', 0.03)
                pgd_alpha = getattr(config, 'smc_pgd_alpha', 0.01)
                pgd_steps = getattr(config, 'smc_pgd_steps', 10)

                particles_adv = particles.clone().detach()
                particles_orig = particles.clone().detach()

                for _ in range(pgd_steps):
                    particles_adv.requires_grad = True
                    with torch.enable_grad():
                        score_output = score_model(particles_adv, current_t)

                        loss_adv = -torch.mean(torch.sum(score_output ** 2, dim=-1))

                    grad = torch.autograd.grad(loss_adv, [particles_adv],
                                               retain_graph=False, create_graph=False)[0]

                    particles_adv = particles_adv.detach() - pgd_alpha * grad.sign()
                    delta = torch.clamp(particles_adv - particles_orig, min=-pgd_eps, max=pgd_eps)
                    particles = torch.clamp(particles_orig + delta, min=-15, max=15).detach()

            elif 'fgsm' in perturbation_mode:
                particles_adv = particles.clone().detach()
                particles_adv.requires_grad = True
                fgsm_eps = getattr(config, 'smc_fgsm_eps', 0.01)

                with torch.enable_grad():
                    score_output = score_model(particles_adv, current_t)
                    loss_adv = -torch.mean(torch.sum(score_output ** 2, dim=-1))

                grad = torch.autograd.grad(loss_adv, [particles_adv])[0]
                particles = particles - fgsm_eps * grad.sign()
                particles = torch.clamp(particles, min=-15, max=15).detach()

        if noise_std_per_epoch > 0:
            noise = torch.randn_like(particles) * noise_std_per_epoch
            particles += noise

        epoch += 1
        if p_anneal >= 1.0:
            break

    return particles
