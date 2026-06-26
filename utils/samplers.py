import torch
from tqdm import tqdm


def get_sampler(config, device, sde):
    torch.manual_seed(123)

    def get_euler_maruyama(model):
        x_t = sde.prior_sampling((config.sampling_batch_size,config.dimension),device=device)
        time_pts = sde.time_steps(config.disc_steps, device)
        pbar = tqdm(range(len(time_pts) - 1),leave=False)
        T = sde.T()
        for i in pbar:
            t = time_pts[i]
            dt = time_pts[i + 1] - t
            score = model(x_t, T- t)
            diffusion = sde.diffusion(x_t,T - t)
            tot_drift = - sde.drift(x_t,T - t) + diffusion**2 * score
            x_t += tot_drift * dt + diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5
        pbar.close()
        return x_t



































    def get_exponential_integrator_ADDDS(model):

        x_t_current = sde.prior_sampling((config.sampling_batch_size, config.dimension), device=device)

        time_pts = sde.time_steps(config.disc_steps, device)
        T_sde = sde.T()
        pbar = tqdm(range(len(time_pts) - 1), leave=False)

        for i in pbar:
            t_step = time_pts[i]
            dt_step = time_pts[i + 1] - t_step
            e_h = torch.exp(dt_step)
            x_t_noise = x_t_current.clone()

            perturbation_mode = getattr(config, 'perturbation_mode', 'gaussian').lower()
            if 'pgd' in perturbation_mode:
                pgd_eps = getattr(config, 'pgd_eps', 0.03)
                pgd_alpha = getattr(config, 'pgd_alpha', 0.01)
                pgd_steps = getattr(config, 'pgd_steps', 10)
                random_start = getattr(config, 'pgd_random_start', True)

                x_t_adv = x_t_current.clone().detach()
                x_t_orig = x_t_current.clone().detach()

                if random_start:
                    x_t_adv = x_t_adv + torch.empty_like(x_t_adv).uniform_(-pgd_eps, pgd_eps)
                    x_t_adv = torch.clamp(x_t_adv, min=-15, max=15).detach()

                for _ in range(pgd_steps):
                    x_t_adv.requires_grad = True
                    with torch.enable_grad():
                        score_output = model(x_t_adv, T_sde - t_step)
                        loss_adv = -torch.mean(torch.sum(score_output ** 2, dim=-1))

                    grad = torch.autograd.grad(loss_adv, [x_t_adv],
                                               retain_graph=False, create_graph=False)[0]

                    x_t_adv = x_t_adv.detach() - pgd_alpha * grad.sign()
                    delta = torch.clamp(x_t_adv - x_t_orig, min=-pgd_eps, max=pgd_eps)
                    x_t_adv = torch.clamp(x_t_orig + delta, min=-15, max=15).detach()

                x_t_noise = x_t_adv
            elif 'fgsm' in perturbation_mode:
                x_t_adv = x_t_current.clone().detach()
                x_t_adv.requires_grad = True
                fgsm_eps = getattr(config, 'fgsm_eps', 0.01)

                with torch.enable_grad():
                    score_output = model(x_t_adv, T_sde - t_step)
                    loss_adv = -torch.mean(torch.sum(score_output ** 2, dim=-1))

                grad = torch.autograd.grad(loss_adv, [x_t_adv])[0]

                x_t_adv = x_t_adv - fgsm_eps * grad.sign()
                x_t_noise = torch.clamp(x_t_adv, min=-15, max=15).detach()


            elif 'gcg' in perturbation_mode:
                gcg_eps = getattr(config, 'gcg_eps', 0.03)
                gcg_alpha = getattr(config, 'gcg_alpha', 0.01)
                gcg_steps = getattr(config, 'gcg_steps', 10)
                gcg_k = getattr(config, 'gcg_k', 1)
                random_start = getattr(config, 'pgd_random_start', True)

                x_t_adv = x_t_current.clone().detach()
                x_t_orig = x_t_current.clone().detach()

                if random_start:
                    x_t_adv = x_t_adv + torch.empty_like(x_t_adv).uniform_(-gcg_eps, gcg_eps)
                    x_t_adv = torch.clamp(x_t_adv, min=-15, max=15).detach()

                for _ in range(gcg_steps):
                    x_t_adv.requires_grad = True
                    with torch.enable_grad():
                        score_output = model(x_t_adv, T_sde - t_step)
                        loss_adv = -torch.mean(torch.sum(score_output ** 2, dim=-1))

                    grad = torch.autograd.grad(loss_adv, [x_t_adv],
                                               retain_graph=False, create_graph=False)[0]



                    k = min(gcg_k, grad.shape[-1])
                    _, topk_indices = torch.topk(torch.abs(grad), k=k, dim=-1)


                    mask = torch.zeros_like(grad).scatter_(-1, topk_indices, 1.0)


                    x_t_adv = x_t_adv.detach() - gcg_alpha * grad.sign() * mask


                    delta = torch.clamp(x_t_adv - x_t_orig, min=-gcg_eps, max=gcg_eps)
                    x_t_adv = torch.clamp(x_t_orig + delta, min=-15, max=15).detach()

                x_t_noise = x_t_adv


            elif 'gaussian' in perturbation_mode:
                gaussian_perturbation = config.added_noise_std_for_reg * torch.randn_like(x_t_current)
                x_t_noise = x_t_current + gaussian_perturbation


            model_kwargs = {}
            perturbation = torch.abs(x_t_noise - x_t_current)
            reg_val_scalar = torch.mean(torch.sum(perturbation, dim=-1))
            model_kwargs['reg_val_scalar_for_model'] = reg_val_scalar

            score_val = model(x_t_noise, T_sde - t_step, **model_kwargs)
            x_t_current = e_h * x_t_current + 2 * (e_h - 1) * score_val + ((e_h ** 2 - 1)) ** .5 * torch.randn_like(
                x_t_current)

        pbar.close()
        return x_t_current

























    def get_exponential_integrator(model):

        x_t = sde.prior_sampling((config.sampling_batch_size, config.dimension), device=device)

        time_pts = sde.time_steps(config.disc_steps, device)
        T = sde.T()
        pbar = tqdm(range(len(time_pts) - 1), leave=False)
        for i in pbar:
            t = time_pts[i]
            dt = time_pts[i + 1] - t


            if getattr(config, 'use_pc_sampler', False):


                grad = model(x_t, T - t)
                noise = torch.randn_like(x_t)


                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (config.pc_snr * noise_norm / grad_norm) ** 2 * 2 * 1.0

                x_mean = x_t + step_size * grad
                x_t = x_mean + torch.sqrt(step_size * 2) * noise



            score = model(x_t, T - t)



            if getattr(config, 'use_score_clipping', False):
                clip_norm = getattr(config, 'score_clip_norm', 10.0)
                score_norm = torch.norm(score, dim=-1, keepdim=True)

                scale_factor = torch.minimum(torch.ones_like(score_norm), clip_norm / (score_norm + 1e-6))
                score = score * scale_factor

            e_h = torch.exp(dt)
            x_t_pure = e_h * x_t + 2 * (e_h - 1) * score + ((e_h ** 2 - 1)) ** .5 * torch.randn_like(x_t)
            perturbation_mode = getattr(config, 'perturbation_mode', 'gaussian').lower()
            if 'pgd' in perturbation_mode:
                pgd_eps = getattr(config, 'pgd_eps', 0.03)
                pgd_alpha = getattr(config, 'pgd_alpha', 0.01)
                pgd_steps = getattr(config, 'pgd_steps', 10)
                random_start = getattr(config, 'pgd_random_start', True)

                x_t_adv = x_t.clone().detach()
                x_t_orig = x_t.clone().detach()
                if random_start:
                    x_t_adv += torch.empty_like(x_t_adv).uniform_(-pgd_eps, pgd_eps)
                    x_t_adv = torch.clamp(x_t_adv, min=-15, max=15).detach()

                for _ in range(pgd_steps):
                    x_t_adv.requires_grad = True
                    with torch.enable_grad():
                        score_output = model(x_t_adv, T - t)
                        loss_adv = -torch.mean(torch.sum(score_output ** 2, dim=-1))

                    grad = torch.autograd.grad(loss_adv, [x_t_adv], retain_graph=False, create_graph=False)[0]

                    x_t_adv = x_t_adv.detach() - pgd_alpha * grad.sign()
                    delta = torch.clamp(x_t_adv - x_t_orig, min=-pgd_eps, max=pgd_eps)
                    x_t_adv = torch.clamp(x_t_orig + delta, min=-15, max=15).detach()

                x_t_perturbed = x_t_adv
            elif 'fgsm' in perturbation_mode:
                x_t_adv = x_t.clone().detach().requires_grad_(True)
                fgsm_eps = getattr(config, 'fgsm_eps', 0.01)

                with torch.enable_grad():
                    score_output = model(x_t_adv, T - t)
                    loss_adv = -torch.mean(torch.sum(score_output ** 2, dim=-1))

                grad = torch.autograd.grad(loss_adv, [x_t_adv])[0]

                x_t_perturbed = x_t - fgsm_eps * grad.sign()
                x_t_perturbed = torch.clamp(x_t_perturbed, min=-15, max=15).detach()


            elif 'gcg' in perturbation_mode:
                gcg_eps = getattr(config, 'gcg_eps', 0.03)
                gcg_alpha = getattr(config, 'gcg_alpha', 0.01)
                gcg_steps = getattr(config, 'gcg_steps', 10)
                gcg_k = getattr(config, 'gcg_k', 1)
                random_start = getattr(config, 'pgd_random_start', True)

                x_t_adv = x_t.clone().detach()
                x_t_orig = x_t.clone().detach()

                if random_start:
                    x_t_adv += torch.empty_like(x_t_adv).uniform_(-gcg_eps, gcg_eps)
                    x_t_adv = torch.clamp(x_t_adv, min=-15, max=15).detach()

                for _ in range(gcg_steps):
                    x_t_adv.requires_grad = True
                    with torch.enable_grad():
                        score_output = model(x_t_adv, T - t)
                        loss_adv = -torch.mean(torch.sum(score_output ** 2, dim=-1))

                    grad = torch.autograd.grad(loss_adv, [x_t_adv], retain_graph=False, create_graph=False)[0]


                    k = min(gcg_k, grad.shape[-1])
                    _, topk_indices = torch.topk(torch.abs(grad), k=k, dim=-1)
                    mask = torch.zeros_like(grad).scatter_(-1, topk_indices, 1.0)


                    x_t_adv = x_t_adv.detach() - gcg_alpha * grad.sign() * mask


                    delta = torch.clamp(x_t_adv - x_t_orig, min=-gcg_eps, max=gcg_eps)
                    x_t_adv = torch.clamp(x_t_orig + delta, min=-15, max=15).detach()

                x_t_perturbed = x_t_adv


            elif 'gaussian' in perturbation_mode:
                gaussian_perturbation = config.added_noise_std_for_reg * torch.randn_like(x_t)
                x_t_perturbed = x_t + gaussian_perturbation

        pbar.close()
        return x_t_perturbed




    if config.sampling_method == 'em':
        return get_euler_maruyama
    if config.sampling_method == 'ei':
        return get_exponential_integrator
    if config.sampling_method == 'ei_ADDDS':
        return get_exponential_integrator_ADDDS
