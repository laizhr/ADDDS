import torch
from tqdm import tqdm
from utils.densities import Distribution


def parallel_tempering(distribution: Distribution,
                       initial_cond, betas, num_iters, h, device='cuda', added_noise_std=0.0, config=None):

    d = distribution.dim
    num_chains = betas.shape[0]
    betas = betas.reshape(num_chains, 1, 1)
    xchains = initial_cond.expand(num_chains, *initial_cond.shape).clone().to(device) if len(
        initial_cond.shape) == 2 else initial_cond


    perform_attack = config is not None
    if perform_attack:
        perturbation_mode = getattr(config, 'perturbation_mode', 'none').lower()
    else:
        perturbation_mode = 'none'

    for i in tqdm(range(num_iters), leave=False):

        xchains_attacked = xchains.clone().detach()

        if 'pgd' in perturbation_mode:
            pgd_eps = getattr(config, 'pgd_eps', 0.03)
            pgd_alpha = getattr(config, 'pgd_alpha', 0.01)
            pgd_steps = getattr(config, 'pgd_steps', 10)

            xchains_adv = xchains.clone().detach()
            xchains_orig = xchains.clone().detach()

            for _ in range(pgd_steps):
                xchains_adv.requires_grad = True
                with torch.enable_grad():
                    loss_adv = -torch.mean(torch.sum(distribution.grad_log_prob(xchains_adv) ** 2, dim=-1))
                grad = torch.autograd.grad(loss_adv, [xchains_adv],
                                           retain_graph=False, create_graph=False)[0]
                xchains_adv = xchains_adv.detach() - pgd_alpha * grad.sign()
                delta = torch.clamp(xchains_adv - xchains_orig, min=-pgd_eps, max=pgd_eps)
                xchains_attacked = torch.clamp(xchains_orig + delta, min=-15, max=15).detach()

        elif 'fgsm' in perturbation_mode:
            xchains_adv = xchains.clone().detach()
            xchains_adv.requires_grad = True
            fgsm_eps = getattr(config, 'fgsm_eps', 0.01)

            with torch.enable_grad():
                loss_adv = -torch.mean(torch.sum(distribution.grad_log_prob(xchains_adv) ** 2, dim=-1))

            grad = torch.autograd.grad(loss_adv, [xchains_adv])[0]
            xchains_attacked = xchains_adv - fgsm_eps * grad.sign()
            xchains_attacked = torch.clamp(xchains_attacked, min=-15, max=15).detach()

        elif 'gaussian' in perturbation_mode:
            gaussian_perturbation = config.added_noise_std_for_reg * torch.randn_like(xchains)
            xchains_attacked = xchains + gaussian_perturbation

        xk = xchains_attacked
        center = xk + h * betas * distribution.grad_log_prob(xk).nan_to_num()
        proposal = center + (2 * h) ** .5 * torch.randn_like(xk)
        center_proposal = proposal + h * betas * distribution.grad_log_prob(proposal)

        prob1 = distribution.log_prob(proposal) - torch.sum((proposal - center) ** 2, dim=-1, keepdim=True) / (4 * h)
        prob2 = distribution.log_prob(xk) - torch.sum((xk - center_proposal) ** 2, dim=-1, keepdim=True) / (4 * h)

        acc_rate = torch.exp(prob1 - prob2)
        acc_rate = torch.min(torch.ones_like(acc_rate), acc_rate)
        acc = torch.rand_like(acc_rate) < acc_rate
        acc = acc.expand((-1, -1, d))

        xchains = torch.where(acc, proposal, xk)
        for k in range(1, num_chains):
            xii = xchains[k - 1]
            xi = xchains[k]
            acc_rate = torch.exp((betas[k] - betas[k - 1]) * distribution.log_prob(xii)
                                 + (betas[k - 1] - betas[k]) * distribution.log_prob(xi))
            acc_rate = torch.min(torch.ones_like(acc_rate), acc_rate)
            acc = torch.rand_like(acc_rate) < acc_rate
            acc = acc.expand((-1, d))


            temp_xi = xi.clone()
            xchains[k - 1][acc] = temp_xi[acc]
            xchains[k][acc] = xii[acc]

        if added_noise_std > 0:
            xchains += torch.randn_like(xchains) * added_noise_std

    return xchains[num_chains - 1], xchains