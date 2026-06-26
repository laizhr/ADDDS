import torch
from tqdm import tqdm
from utils.densities import Distribution
from utils.optimizers import nesterovs_minimizer


def sum_last_dim(x):
    return torch.sum(x, dim=-1, keepdim=True)


def get_rgo_sampling(xk, yk, eta, dist: Distribution, M, device, initial_cond_for_minimization=None):
    num_samples, d = xk.shape
    al, delta = 1, 1
    accepted_samples = torch.ones_like(xk)
    num_acc_samples = 0
    f_eta_pot = lambda x: -dist.log_prob(x) + sum_last_dim((x - yk) ** 2) / (2 * eta)
    grad_f_eta = lambda x: -dist.grad_log_prob(x) + (x - yk) / eta
    in_cond = xk if initial_cond_for_minimization == None else initial_cond_for_minimization
    w, k = nesterovs_minimizer(in_cond, grad_f_eta, eta, M)
    min_val = dist.log_prob(w)
    var = 1 / (1 / eta - M)
    gradw = -dist.grad_log_prob(w)
    u = (yk / eta - gradw - M * w) * var
    num_rejection_iters = 0
    while num_acc_samples < num_samples * d and num_rejection_iters < 20:
        num_rejection_iters += 1
        proposal = u + var ** .5 * accepted_samples * torch.randn_like(xk)
        exp_h1 = - min_val\
                 + sum_last_dim(gradw * (proposal - w))\
                 - M * sum_last_dim((proposal - w) ** 2) / 2\
                 - (1 - al) * delta / 2
        f_eta = -dist.log_prob(proposal)
        rand_prob = torch.rand((num_samples, 1), device=device)
        acc_idx = (accepted_samples * torch.exp(-exp_h1) * rand_prob <= torch.exp(-f_eta))
        num_acc_samples = torch.sum(acc_idx)
        accepted_samples = (~acc_idx).long()
        u[acc_idx] = proposal[acc_idx]
    xk[acc_idx] = proposal[acc_idx]

    return xk, 1 + num_rejection_iters + k, w


def get_samples(x0, dist: Distribution, M, num_iters, num_samples, device, max_grad_complexity=None,
                added_noise_std=0.0, config=None):

    n, d = x0.shape[0], x0.shape[-1]
    xk = x0.repeat_interleave(num_samples, dim=0)
    w = None
    eta = 1 / (M * d)
    tot_grad_complexity = 0
    perform_attack = config is not None
    if perform_attack:
        perturbation_mode = getattr(config, 'perturbation_mode', 'none').lower()
    else:
        perturbation_mode = 'none'

    for _ in tqdm(range(num_iters), leave=False):

        xk_attacked = xk.clone().detach()
        if 'pgd' in perturbation_mode:
            pgd_eps = getattr(config, 'pgd_eps', 0.03)
            pgd_alpha = getattr(config, 'pgd_alpha', 0.01)
            pgd_steps = getattr(config, 'pgd_steps', 10)

            xk_adv = xk.clone().detach()
            xk_orig = xk.clone().detach()

            for _ in range(pgd_steps):
                xk_adv.requires_grad = True
                with torch.enable_grad():

                    loss_adv = -torch.mean(torch.sum(dist.grad_log_prob(xk_adv) ** 2, dim=-1))

                grad = torch.autograd.grad(loss_adv, [xk_adv],
                                           retain_graph=False, create_graph=False)[0]

                xk_adv = xk_adv.detach() - pgd_alpha * grad.sign()
                delta = torch.clamp(xk_adv - xk_orig, min=-pgd_eps, max=pgd_eps)
                xk_attacked = torch.clamp(xk_orig + delta, min=-15, max=15).detach()

        elif 'fgsm' in perturbation_mode:
            xk_adv = xk.clone().detach()
            xk_adv.requires_grad = True
            fgsm_eps = getattr(config, 'fgsm_eps', 0.01)

            with torch.enable_grad():
                loss_adv = -torch.mean(torch.sum(dist.grad_log_prob(xk_adv) ** 2, dim=-1))

            grad = torch.autograd.grad(loss_adv, [xk_adv])[0]
            xk_attacked = xk_adv - fgsm_eps * grad.sign()
            xk_attacked = torch.clamp(xk_attacked, min=-15, max=15).detach()

        elif 'gaussian' in perturbation_mode:
            gaussian_perturbation = config.added_noise_std_for_reg * torch.randn_like(xk)
            xk_attacked = xk + gaussian_perturbation

        z = torch.randn_like(xk_attacked, device=device)
        yk = xk_attacked + z * eta ** .5
        xk, _, w = get_rgo_sampling(xk_attacked, yk, eta, dist, M, device, w)

        if added_noise_std > 0:
            xk += torch.randn_like(xk) * added_noise_std



    return xk.reshape((n, num_samples, -1))