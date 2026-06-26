import os
import torch
import numpy as np
import samplers.parallel_tempering
import sde_lib
import utils.plots
import utils.densities
import utils.metrics
import sample
import matplotlib.pyplot as plt
import samplers.ula
import samplers.proximal_sampler
from samplers.SMC import smc_sampler_for_zodmc
from utils.gmm_score import get_gmm_density_at_t

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_num_methods(config):
    num = len(config.methods_to_run) + len(config.baselines)
    num += 1 if config.eval_mmd else 0
    return num


def apply_perturbation(x, score_fn, t, config):
    x_curr = x.clone().detach()
    x_adv = x_curr.clone().detach()

    mode = config.perturbation_mode.lower()

    if 'pgd' in mode:
        if getattr(config, 'pgd_random_start', True):
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-config.pgd_eps, config.pgd_eps)
            x_adv = torch.clamp(x_adv, min=-15, max=15).detach()

        for _ in range(config.pgd_steps):
            x_adv.requires_grad = True
            with torch.enable_grad():
                score_out = score_fn(x_adv, t)
                loss = -torch.mean(torch.sum(score_out ** 2, dim=-1))

            grad = torch.autograd.grad(loss, [x_adv], retain_graph=False, create_graph=False)[0]
            x_adv = x_adv.detach() - config.pgd_alpha * grad.sign()

            delta = torch.clamp(x_adv - x_curr, min=-config.pgd_eps, max=config.pgd_eps)
            x_adv = torch.clamp(x_curr + delta, min=-15, max=15).detach()

        x_perturbed = x_adv

    elif 'fgsm' in mode:

        x_adv.requires_grad = True
        with torch.enable_grad():
            score_out = score_fn(x_adv, t)
            loss = -torch.mean(torch.sum(score_out ** 2, dim=-1))

        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_perturbed = x_adv - config.fgsm_eps * grad.sign()
        x_perturbed = torch.clamp(x_perturbed, min=-15, max=15).detach()

    elif 'gcg' in mode:
        gcg_eps = getattr(config, 'gcg_eps', 0.5)
        gcg_alpha = getattr(config, 'gcg_alpha', 0.01)
        gcg_steps = getattr(config, 'gcg_steps', 10)
        gcg_k = getattr(config, 'gcg_k', 1)

        if getattr(config, 'pgd_random_start', True):
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-gcg_eps, gcg_eps)
            x_adv = torch.clamp(x_adv, min=-15, max=15).detach()

        for _ in range(gcg_steps):
            x_adv.requires_grad = True
            with torch.enable_grad():
                score_out = score_fn(x_adv, t)
                loss = -torch.mean(torch.sum(score_out ** 2, dim=-1))

            grad = torch.autograd.grad(loss, [x_adv], retain_graph=False, create_graph=False)[0]

            k = min(gcg_k, grad.shape[-1])
            _, topk_indices = torch.topk(torch.abs(grad), k=k, dim=-1)
            mask = torch.zeros_like(grad).scatter_(-1, topk_indices, 1.0)
            x_adv = x_adv.detach() - gcg_alpha * grad.sign() * mask
            delta = torch.clamp(x_adv - x_curr, min=-gcg_eps, max=gcg_eps)
            x_adv = torch.clamp(x_curr + delta, min=-15, max=15).detach()

        x_perturbed = x_adv
    elif 'gaussian' in mode:

        noise = config.added_noise_std_for_reg * torch.randn_like(x_curr)
        x_perturbed = x_curr + noise

    else:
        x_perturbed = x_curr

    perturbation_mag = torch.abs(x_perturbed - x_curr)
    reg_val = torch.mean(torch.sum(perturbation_mag, dim=-1))

    return x_perturbed, reg_val


def compute_score_mse(config, sde, score_fn, device, num_samples=1000, num_time_steps=10):

    if config.density not in ['gmm']:
        return 0.0, 0.0

    mse_list = []
    ts = torch.linspace(1e-2, config.T - 1e-2, num_time_steps, device=device)

    for t in ts:
        _, true_grad_fn = get_gmm_density_at_t(config, sde, t, device)
        with torch.no_grad():
            dist0 = utils.densities.get_distribution(config, device)
            x0 = dist0.sample(num_samples)
            scale = sde.scaling(t)
            z = torch.randn_like(x0)
            x_t_clean = x0 * scale + torch.sqrt(1 - scale**2) * z

        x_t_perturbed, reg_val = apply_perturbation(x_t_clean, score_fn, t, config)

        with torch.no_grad():
            true_score = true_grad_fn(x_t_perturbed)
            kwargs = {}
            if hasattr(config, 'optimizer_reg') and str(config.optimizer_reg) == 'True':
                 kwargs['reg_val_scalar_for_model'] = reg_val

            estimated_score = score_fn(x_t_perturbed, t, **kwargs)
            loss = torch.sum((true_score - estimated_score)**2, dim=-1).mean()
            mse_list.append(loss.item())

    return np.mean(mse_list), np.std(mse_list)


def eval(config):
    set_seed(12)
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    distribution = utils.densities.get_distribution(config,device)
    mmd = utils.metrics.MMDLoss()
    eval_stats = config.eval_mmd
    dim = distribution.dim

    tot_samples = config.num_batches * config.sampling_batch_size
    num_methods = get_num_methods(config)
    method_names = [''] * num_methods
    oracle_complexity = config.num_samples_for_rdmc * np.arange(config.min_num_iters_rdmc,
                                                                config.max_num_iters_rdmc,
                                                                step=config.iters_rdmc_step)
    print("Oracle Complexities:", oracle_complexity)
    samples_all_methods = torch.zeros((num_methods,len(oracle_complexity), tot_samples,dim),dtype=torch.float32, device=device)

    mmd_stats = np.zeros((num_methods, *oracle_complexity.shape),dtype='double')
    w2_stats = np.zeros((num_methods, *oracle_complexity.shape),dtype='double')

    score_error_stats = np.zeros((num_methods, *oracle_complexity.shape), dtype='double')
    score_error_std_stats = np.zeros((num_methods, *oracle_complexity.shape), dtype='double')

    memory_stats = np.zeros((num_methods, *oracle_complexity.shape), dtype='double')

    k = 0
    if eval_stats:
        real_samples = distribution.sample(tot_samples)
        method_names[0] = 'Ground Truth'
        for i in range(len(oracle_complexity)):
            samples_all_methods[0][i] = real_samples
        k+=1

    folder = os.path.dirname(config.save_folder)
    os.makedirs(folder, exist_ok=True)

    smc_beta_proposal_scale = getattr(config, 'smc_beta_proposal', 0.2)

    sde = sde_lib.get_sde(config)
    model = utils.score_estimators.get_score_function(config, distribution, sde, device)

    if not config.load_from_ckpt:
        for method in config.methods_to_run:
            method_names[k] = method
            for i, gc in enumerate(oracle_complexity):
                print(f"Running {method}, Oracle Complexity: {gc}")
                if method == 'ADDDS':
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection'
                    config.optimizer_reg = 'True'
                    config.sampling_method = 'ei_ADDDS'
                    config.sampling_eps = config.sampling_eps_rejec
                    config.num_estimator_batches = 10
                    config.num_estimator_samples = gc//config.num_estimator_batches
                elif method == 'Hessian-ADDDS':
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection_hessian'
                    config.optimizer_reg_lambda = 0.01
                    config.sampling_method = 'ei_ADDDS'
                    config.sampling_eps = config.sampling_eps_rejec
                    config.num_estimator_batches = 10
                    config.num_estimator_samples = gc // config.num_estimator_batches
                elif method == 'ZOD-MC':
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection'
                    config.optimizer_reg = 'False'
                    config.sampling_method = 'ei'
                    config.sampling_eps = config.sampling_eps_rejec
                    config.num_estimator_batches = 10
                    config.num_estimator_samples = gc // config.num_estimator_batches
                elif method == 'RDMC':
                    config.score_method = 'p0t'
                    config.p0t_method = 'ula'
                    config.sampling_method = 'ei'
                    config.num_estimator_batches = 1
                    config.sampling_eps = config.sampling_eps_rdmc
                    config.num_estimator_samples = config.num_samples_for_rdmc
                    config.num_sampler_iterations = gc//config.num_estimator_samples
                    config.initial_cond_type = 'delta'
                elif method == 'RSDMC':
                    config.score_method = 'recursive'
                    config.sampling_method = 'ei'
                    config.num_estimator_batches = 1
                    config.num_recursive_steps = 2
                    config.num_estimator_samples = max(1,int(np.exp(np.log(gc)/(2 * config.num_recursive_steps)))) + 1
                    config.num_sampler_iterations = max(1,int(np.exp(np.log(gc)/(2 * config.num_recursive_steps))))
                elif method == 'SLIPS':
                    config.score_method = 'p0t'
                    config.p0t_method = 'mala'
                    config.num_estimator_batches = 1
                    config.num_estimator_samples = config.num_samples_for_rdmc
                    config.slips_mala_steps = gc // config.num_estimator_samples
                    if config.slips_mala_steps == 0: config.slips_mala_steps = 1
                    config.sampling_eps = getattr(config, 'sampling_eps_slips', config.sampling_eps_rdmc)
                elif method == 'Score_Clipping':
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection'
                    config.optimizer_reg = 'False'
                    config.sampling_method = 'ei'
                    config.use_score_clipping = True
                    config.score_clip_norm = 10.0
                elif method == 'PC_Sampler':
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection'
                    config.optimizer_reg = 'False'
                    config.sampling_method = 'ei'
                    config.use_score_clipping = False
                    config.use_pc_sampler = True
                    config.pc_snr = 0.16

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                samples_all_methods[k][i] = sample.sample(config)
                if config.density == 'gmm':
                    current_score_fn = utils.score_estimators.get_score_function(config, distribution, sde, device)
                    score_err, score_std = compute_score_mse(config, sde, current_score_fn, device, num_samples=1000)

                    score_error_stats[k][i] = score_err
                    score_error_std_stats[k][i] = score_std

                   
                if torch.cuda.is_available():
                    max_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                else:
                    max_mem = 0.0

                memory_stats[k][i] = max_mem

                if eval_stats:
                    mmd_stats[k][i] = mmd.get_mmd_squared(samples_all_methods[k][i],real_samples).detach().item()
                    w2_stats[k][i] = utils.metrics.get_w2(samples_all_methods[k][i],real_samples).detach().item()
        

            k+=1

        for baseline in config.baselines:
            prev = 0
            method_names[k] = baseline
            prev_gc_iter = 0
            in_cond = torch.randn((tot_samples, dim), dtype=torch.float32, device=device)
            parallel_curr_state = None
            for i, gc in enumerate(oracle_complexity):
                gc_step_complexity = config.disc_steps * (gc - prev_gc_iter)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                if baseline == 'langevin':
                    samples_all_methods[k][i] = samplers.ula.get_ula_samples(in_cond,
                                                                    distribution.grad_log_prob,
                                                                    config.langevin_step_size,
                                                                    config.disc_steps * (gc - prev) ,
                                                                    display_pbar=False,
                                                                    added_noise_std=config.added_noise_std_for_reg,
                                                                    config=config)
                elif baseline == 'proximal':
                    samples_all_methods[k][i] = samplers.proximal_sampler.get_samples(in_cond,
                                                                distribution,
                                                                config.proximal_M,
                                                                config.disc_steps * (gc - prev),
                                                                1,
                                                                device,
                                                                max_grad_complexity = config.disc_steps * (gc - prev),
                                                                added_noise_std=config.added_noise_std_for_reg,
                                                                config=config
                                                                ).squeeze(1)
                elif baseline == 'parallel':
                    num_chains = config.num_chains_parallel
                    num_iters = config.disc_steps * (gc - prev)//(6 * num_chains)
                    betas = torch.linspace(.2,1.,num_chains, dtype=torch.float32,device=device)
                    in_cond = in_cond if i == 0 else parallel_curr_state
                    samples_all_methods[k][i], parallel_curr_state = samplers.parallel_tempering.parallel_tempering(distribution,
                                                                                                                    in_cond,betas,
                                                                                                                    num_iters,
                                                                                                                    config.langevin_step_size,
                                                                                                                    device,
                                                                                                                    added_noise_std=config.added_noise_std_for_reg,
                                                                                                                    config=config)
                elif baseline == 'smc':
                    if tot_samples > 0:
                        smc_mcmc_steps_this_gc = max(1, gc_step_complexity // tot_samples)
                    else:
                        smc_mcmc_steps_this_gc = max(1, gc_step_complexity)
                    smc_mcmc_steps_this_gc = max(1, smc_mcmc_steps_this_gc)

                    samples_all_methods[k][i] = smc_sampler_for_zodmc(
                        target_dist=distribution,
                        dim=dim,
                        n_particles=tot_samples,
                        n_mcmc_steps_per_particle=smc_mcmc_steps_this_gc,
                        beta_smc=smc_beta_proposal_scale,
                        device=device,
                        oracle_complexity_info=gc,
                        noise_std_per_epoch = config.added_noise_std_for_reg,
                        config = config,
                        score_model = model
                    )
                else:
                    print(f'The baseline method {baseline} has not been implemented yet')

                if torch.cuda.is_available():
                    memory_stats[k][i] = torch.cuda.max_memory_allocated() / (1024 * 1024)
                else:
                    memory_stats[k][i] = 0.0

                prev = gc
                samples_all_methods[k,i][samples_all_methods[k,i].abs() > 100] = 0.
                in_cond = samples_all_methods[k][i]

                if eval_stats:
                    mmd_stats[k][i] = mmd.get_mmd_squared(samples_all_methods[k][i],real_samples).detach().item()
                    w2_stats[k][i] = utils.metrics.get_w2(samples_all_methods[k][i],real_samples).detach().item()

            k+=1

    else:
        samples_all_methods = torch.load(config.samples_ckpt).to(device=device).to(dtype=torch.float32)
        method_names = np.load(os.path.join(folder,'method_names.npy'))
        mmd_stats = np.zeros((len(method_names), *oracle_complexity.shape),dtype='double')

        if eval_stats:
            for k, method in enumerate(method_names):
                if method == 'Ground Truth':
                    k-=1
                    continue
                for i, gc in enumerate(oracle_complexity):
                    if eval_stats:
                        mmd_stats[k][i] = mmd.get_mmd_squared(samples_all_methods[k][i],real_samples).detach().item()
                        w2_stats[k][i] = utils.metrics.get_w2(samples_all_methods[k][i],real_samples).detach().item()

    save_file = os.path.join(folder,f'samples_{config.density}.pt')
    np.save(os.path.join(folder,'method_names.npy'), np.array(method_names))
    torch.save(samples_all_methods, save_file)

    if dim == 2:
        take_log = config.density not in ['lmm','gmm']
        xlim = [-5,13] if config.density in ['lmm','gmm'] else [-5, 9]
        ylim = [-5,13] if config.density in ['lmm','gmm']else [-8,3.5]
        for i, gc in enumerate(oracle_complexity):
            fig = utils.plots.plot_all_samples(samples_all_methods[:,i,:,:],
                                            method_names,
                                            xlim,ylim,distribution.log_prob,take_log)
            plt.close(fig)
            fig.savefig(os.path.join(folder,f'complexity_{gc}_{config.density}.png'), bbox_inches='tight')
    else:
        take_log = config.density not in ['lmm','gmm']
        xlim = [-13,13] if config.density in ['lmm','gmm'] else [-5, 9]
        ylim = [-13,13] if config.density in ['lmm','gmm']else [-8,3.5]
        for i, gc in enumerate(oracle_complexity):
            fig = utils.plots.plot_all_samples(samples_all_methods[:,i,:,:],
                                            method_names,
                                            xlim,ylim,None,take_log)
            plt.close(fig)
            fig.savefig(os.path.join(folder,f'complexity_{gc}_{config.density}.pdf'), bbox_inches='tight')

    if eval_stats:
        plt.rcParams.update({'font.size': 20})
        ls=['--','-.',':']
        markers=['p','*','s','d','h']

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
        for i, method in enumerate(method_names):
            if method == 'Ground Truth':
                continue
            ax1.plot(oracle_complexity,mmd_stats[i],label=method,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
            ax2.plot(oracle_complexity,w2_stats[i],label=method,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
        ax1.set_xlabel('Oracle Complexity')
        ax1.set_ylabel('MMD')
        ax1.legend(loc='upper left',bbox_to_anchor=(0.6,0.8))
        ax2.set_xlabel('Oracle Complexity')
        ax2.set_ylabel('W2')
        ax2.legend(loc='upper left',bbox_to_anchor=(0.6,0.6))
        fig.savefig(os.path.join(folder,f'mmd_results_{dim}_{config.density}.pdf'),bbox_inches='tight')
        plt.close(fig)

        if config.density == 'gmm':
            plt.figure(figsize=(10, 8))
            ax_err = plt.gca()
            for i, method in enumerate(method_names):
                if method == 'Ground Truth' or method == '': continue
                ax_err.errorbar(oracle_complexity, score_error_stats[i], yerr=score_error_std_stats[i],
                                label=method, linestyle=ls[i%3], marker=markers[i%5], markersize=7, capsize=5)
            ax_err.set_xlabel('Oracle Complexity', fontsize=25)
            ax_err.set_ylabel('Score Error (MSE)', fontsize=25)
            ax_err.legend(loc='upper right', fontsize=20)
            ax_err.grid(True, which="both", ls="--", alpha=0.5)
            plt.savefig(os.path.join(folder, f'score_error_results_{dim}_{config.density}.pdf'), bbox_inches='tight')
            plt.close()

    line_styles = ['-','--', '-.', ':', '-', '--', '-.', ':']
    marker_styles = ['o', 's', 'D', '^', 'v', 'p', '*', 'h', '+']

    plt.figure(figsize=(10, 8))
    ax_mem = plt.gca()
    for i, method in enumerate(method_names):
        if method == 'Ground Truth' or method == '': continue
        ax_mem.plot(oracle_complexity, memory_stats[i], label=method,
                    linestyle=line_styles[i % len(line_styles)],
                    marker=marker_styles[i % len(marker_styles)],
                    markersize=7)
    ax_mem.set_xlabel('Oracle Complexity',fontsize=25)
    ax_mem.set_ylabel('Peak Memory (MB)',fontsize=25)
    ax_mem.legend(loc='upper left',fontsize=20)
    ax_mem.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(folder, f'memory_overhead_{dim}_{config.density}.pdf'), bbox_inches='tight')
    plt.close()

    print(f"All plots saved to {folder}")
