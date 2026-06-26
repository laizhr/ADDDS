import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import click
from types import SimpleNamespace

import utils.score_estimators as score_estimators
from sde_lib import VP
from utils.gmm_score import get_gmm_density_at_t_no_config


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def to_tensor_type(x, device):
    return torch.tensor(x, device=device, dtype=torch.float32)


def get_gmm(path, device):
    params = yaml.safe_load(open(path))
    c = to_tensor_type(params['coeffs'], device)
    means = to_tensor_type(params['means'], device)
    variances = to_tensor_type(params['variances'], device)
    return c, means, variances


def get_l2_error(real_score, generated_score):
    errors = torch.sum((real_score - generated_score) ** 2, dim=-1) ** .5
    mean_error = torch.mean(errors)
    std = torch.mean((errors - mean_error) ** 2) ** .5
    return mean_error, std


def apply_perturbation(samples, model, t, T_sde, config):
    x_t_current = samples.clone().detach()
    x_t_noise = x_t_current.clone()

    perturbation_mode = config.perturbation_mode.lower()

    if 'pgd' in perturbation_mode:
        x_t_adv = x_t_current.clone().detach()
        x_t_orig = x_t_current.clone().detach()
        x_t_adv = x_t_adv + torch.empty_like(x_t_adv).uniform_(-config.pgd_eps, config.pgd_eps)
        x_t_adv = torch.clamp(x_t_adv, min=-15, max=15).detach()

        for _ in range(config.pgd_steps):
            x_t_adv.requires_grad = True
            with torch.enable_grad():
                score_output = model(x_t_adv, T_sde - t)
                loss_adv = -torch.mean(torch.sum(torch.abs(score_output), dim=-1))

            grad = torch.autograd.grad(loss_adv, [x_t_adv],
                                       retain_graph=False, create_graph=False)[0]

            if torch.isnan(grad).any():
                break

            x_t_adv = x_t_adv.detach() - config.pgd_alpha * grad.sign()
            delta = torch.clamp(x_t_adv - x_t_orig, min=-config.pgd_eps, max=config.pgd_eps)
            x_t_noise = torch.clamp(x_t_orig + delta, min=-15, max=15).detach()

    elif 'fgsm' in perturbation_mode:
        x_t_adv = x_t_current.clone().detach()
        x_t_adv.requires_grad = True
        with torch.enable_grad():
            score_output = model(x_t_adv, T_sde - t)
            loss_adv = -torch.mean(torch.sum(torch.abs(score_output), dim=-1))
        grad = torch.autograd.grad(loss_adv, [x_t_adv])[0]
        if not torch.isnan(grad).any():
            x_t_adv = x_t_adv - config.fgsm_eps * grad.sign()
            x_t_noise = torch.clamp(x_t_adv, min=-15, max=15).detach()

    elif 'gaussian' in perturbation_mode:
        gaussian_perturbation = config.added_noise_std_for_reg * torch.randn_like(x_t_current)
        x_t_noise = x_t_current + gaussian_perturbation

    perturbation = torch.abs(x_t_noise - x_t_current)
    reg_val_scalar = torch.mean(torch.sum(perturbation, dim=-1))

    return x_t_noise, reg_val_scalar


@click.command()
@click.option('--num_samples_pt', type=int, default=1000)
@click.option('--save_folder', type=str)
@click.option('--density_params_path', type=str)
@click.option('--load_from_ckpt', is_flag=True)
@click.option('--perturbation_mode', type=click.Choice(['gaussian', 'fgsm', 'pgd']), default='gaussian',
              help='Type of perturbation to apply.')
@click.option('--added_noise_std_for_reg', type=float, default=0.01, help='Std dev for Gaussian noise perturbation.')
@click.option('--fgsm_eps', type=float, default=0.01, help='Epsilon for FGSM attack.')
@click.option('--pgd_eps', type=float, default=0.03, help='Epsilon for PGD attack (radius).')
@click.option('--pgd_alpha', type=float, default=0.01, help='Step size for PGD attack.')
@click.option('--pgd_steps', type=int, default=10, help='Number of steps for PGD attack.')
def eval(num_samples_pt, save_folder, density_params_path, load_from_ckpt,
         perturbation_mode, added_noise_std_for_reg, fgsm_eps, pgd_eps, pgd_alpha, pgd_steps):
    setup_seed(1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config_dict = {
        'perturbation_mode': perturbation_mode,
        'added_noise_std_for_reg': added_noise_std_for_reg,
        'fgsm_eps': fgsm_eps,
        'pgd_eps': pgd_eps,
        'pgd_alpha': pgd_alpha,
        'pgd_steps': pgd_steps,
        'disc_steps': 20
    }
    config = SimpleNamespace(**config_dict)

    folder = os.path.dirname(save_folder)
    os.makedirs(folder, exist_ok=True)

    T = 4.
    delta = .1
    sde = VP(T, delta)
    c, means, variances = get_gmm(density_params_path, device)
    dim = means.shape[-1]

    dist = get_gmm_density_at_t_no_config(sde, torch.tensor([0.], device=device), c, means, variances)
    addds_score_fn = score_estimators.ADDDS(dist, sde, device, config, def_num_rej_samples=1200).score_estimator
    slips_score_fn = score_estimators.SLIPS_ScoreEstimator(dist, sde, device, 1, 1200, 0.1, 5, ).score_estimator
    zodmc_score_fn = score_estimators.ZODMC_ScoreEstimator(dist, sde, device, 10 * dim, 1200).score_estimator
    rdmc_score_normal_fn = score_estimators.RDMC_ScoreEstimator(dist, sde, device, 1, 1200, 0.1, 100, True).score_estimator
    rsdmc_score_fn = score_estimators.RSDMC_ScoreEstimator(dist, sde, device, 1, 10, 0.1, 5, 3, True).score_estimator

    score_fns = [addds_score_fn, zodmc_score_fn, rdmc_score_normal_fn, rsdmc_score_fn, slips_score_fn]
    method_names = ['ADDDS', 'ZOD-MC']

    run_name_suffix = f'perturb_{perturbation_mode}'
    print(f"Running experiment with: {run_name_suffix}")

    num_ts = 20
    num_methods = len(method_names)
    ts = torch.linspace(delta, T, num_ts, device=device)
    errors = torch.zeros((num_methods, num_ts), device=device)
    error_std = torch.zeros((num_methods, num_ts), device=device)

    if not load_from_ckpt:
        for i, t in tqdm(enumerate(ts), total=num_ts):
            dist_t = get_gmm_density_at_t_no_config(sde, t, c, means, variances)
            samples_t = dist_t.sample(num_samples_pt)
            true_score = dist_t.grad_log_prob(samples_t)

            for k, score_fn in enumerate(score_fns):
                perturbed_samples, reg_val_scalar = apply_perturbation(samples_t, addds_score_fn, t, T, config)

                model_kwargs = {}
                if method_names[k] == 'ADDDS':
                    model_kwargs['reg_val_scalar_for_model'] = reg_val_scalar

                estimated_score = score_fn(perturbed_samples, t, **model_kwargs)

                mean, std = get_l2_error(true_score, estimated_score)
                errors[k, i] = mean
                error_std[k, i] = std
    else:
        save_suffix = f'_{perturbation_mode}'
        errors = torch.load(os.path.join(folder, f'errors{save_suffix}.pt'))
        error_std = torch.load(os.path.join(folder, f'std{save_suffix}.pt'))
        method_names = np.load(os.path.join(folder, f'method_names{save_suffix}.npy'))

    save_suffix = f'_{perturbation_mode}'
    torch.save(errors, os.path.join(folder, f'errors{save_suffix}.pt'))
    torch.save(error_std, os.path.join(folder, f'std{save_suffix}.pt'))
    np.save(os.path.join(folder, f'method_names{save_suffix}.npy'), np.array(method_names))

    plt.rcParams.update({
        'font.size': 14,
        'text.usetex': False,
    })
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    ls = ['--', '-.', ':', '-', '-.']
    markers = ['p', '*', 's', 'd', 'h']
    ax1.set_ylim(-1, 4)
    for i, method in enumerate(method_names):
        if method != 'Gaussian':
            ax1.fill_between(ts.cpu(), (errors[i] - error_std[i]).cpu().numpy(),
                             (errors[i] + error_std[i]).cpu().numpy(), alpha=.5)
        ax1.plot(ts.cpu().numpy(), errors[i].cpu().numpy(), label=method, linestyle=ls[i % 3], marker=markers[i % 5],
                 markersize=7)
    ax1.set_xlabel('Time')
    ax1.axhline(y=0, linestyle='dotted', color='black')

    ax1.set_ylabel('E[|| s(x,t) - grad log p(x,t) ||]')
    ax1.legend(loc='upper right')
    fig.savefig(os.path.join(folder, f'error_mmd_results_{dim}{save_suffix}.pdf'), bbox_inches='tight')


if __name__ == '__main__':
    eval()