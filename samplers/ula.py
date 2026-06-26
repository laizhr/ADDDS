import torch
from tqdm import tqdm


def get_ula_samples(xk, grad_log_prob, h, num_iters, display_pbar=True, added_noise_std=0.0, config=None):

    yk = xk.detach().clone()
    perform_attack = config is not None
    if perform_attack:
        perturbation_mode = getattr(config, 'perturbation_mode', 'none').lower()
    else:
        perturbation_mode = 'none'

    for k in tqdm(range(num_iters), leave=False, disable=display_pbar):

        yk_attacked = yk.clone().detach()
        if 'pgd' in perturbation_mode:
            pgd_eps = getattr(config, 'pgd_eps', 0.03)
            pgd_alpha = getattr(config, 'pgd_alpha', 0.01)
            pgd_steps = getattr(config, 'pgd_steps', 10)

            yk_adv = yk.clone().detach()
            yk_orig = yk.clone().detach()

            for _ in range(pgd_steps):
                yk_adv.requires_grad = True
                with torch.enable_grad():
                    loss_adv = -torch.mean(torch.sum(grad_log_prob(yk_adv) ** 2, dim=-1))

                grad = torch.autograd.grad(loss_adv, [yk_adv],
                                           retain_graph=False, create_graph=False)[0]

                yk_adv = yk_adv.detach() - pgd_alpha * grad.sign()
                delta = torch.clamp(yk_adv - yk_orig, min=-pgd_eps, max=pgd_eps)
                yk_attacked = torch.clamp(yk_orig + delta, min=-15, max=15).detach()

        elif 'fgsm' in perturbation_mode:
            yk_adv = yk.clone().detach()
            yk_adv.requires_grad = True
            fgsm_eps = getattr(config, 'fgsm_eps', 0.01)

            with torch.enable_grad():
                loss_adv = -torch.mean(torch.sum(grad_log_prob(yk_adv) ** 2, dim=-1))

            grad = torch.autograd.grad(loss_adv, [yk_adv])[0]
            yk_attacked = yk_adv - fgsm_eps * grad.sign()
            yk_attacked = torch.clamp(yk_attacked, min=-15, max=15).detach()

        elif 'gaussian' in perturbation_mode:
            gaussian_perturbation = config.added_noise_std_for_reg * torch.randn_like(yk)
            yk_attacked = yk + gaussian_perturbation

        yk = yk_attacked + torch.nan_to_num(grad_log_prob(yk_attacked)) * h + (2 * h) ** .5 * torch.randn_like(yk_attacked)

        if added_noise_std > 0:
            yk += torch.randn_like(yk) * added_noise_std

    return yk