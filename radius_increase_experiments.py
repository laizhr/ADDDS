import os
import yaml
import torch
import numpy as np
import utils.plots
import utils.densities
import utils.metrics
import sample
import matplotlib.pyplot as plt
import samplers.ula
import samplers.proximal_sampler
import samplers.SMC as smc_sampler

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def to_tensor_type(x, device):
    return torch.tensor(x,device=device, dtype=torch.float32)

def get_gmm_radius(config,R,device):
    params = yaml.safe_load(open(config.density_parameters_path))
    c = to_tensor_type(params['coeffs'],device)
    means = to_tensor_type(params['means'],device)
    variances = to_tensor_type(params['variances'],device)
    means = R * means / 11
    gaussians = [utils.densities.MultivariateGaussian(means[i],variances[i]) for i in range(c.shape[0])]
    return utils.densities.MixtureDistribution(c,gaussians)

def get_mass_center(config, samples, R):
    dist : utils.densities.MixtureDistribution = get_gmm_radius(config,R,samples.device)
    means = torch.cat([ d.mean.unsqueeze(0) for d in dist.distributions],dim=0).unsqueeze(0)
    idx = torch.argmin(torch.sum((means-samples.view(-1,1,dist.dim))**2,dim=-1),dim=-1)
    return len(idx[idx == 0])/samples.shape[0]

def get_method_names(config):
    num_methods = 1 + len(config.methods_to_run) + len(config.baselines)
    method_names = [''] * num_methods
    method_names[0] = 'Ground Truth'
    k = 1
    for method in config.methods_to_run:
        method_names[k] = method
        k+=1
    for method in config.baselines:
        method_names[k] = method
        k+=1

    return num_methods, method_names


def eval(config):
    setup_seed(1)
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    mmd = utils.metrics.MMDLoss()

    tot_samples = config.num_batches * config.sampling_batch_size
    num_methods, method_names = get_method_names(config)
    radiuses = np.arange(1,17,step=3)
    num_rad = len(radiuses)
    mmd_stats = np.zeros([num_methods, num_rad],dtype='double')
    w2_stats = np.zeros([num_methods, num_rad],dtype='double')
    mass_center = np.zeros_like(w2_stats)

    samples_all = torch.zeros([num_methods, num_rad,tot_samples, config.dimension],device=device,dtype=torch.float32)


    folder = os.path.dirname(config.save_folder)
    os.makedirs(folder, exist_ok=True)

    if not config.load_from_ckpt:
        for i, r in enumerate(radiuses):
            distribution = get_gmm_radius(config,r,device)

            samples_all[0][i] = distribution.sample(tot_samples)
            k = 1
            for method in config.methods_to_run:
                print(method, r)
                if method == 'ADDDS':
                    distribution.keep_minimizer = True
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection'
                    config.optimizer_reg = 'True'
                    config.sampling_method = 'ei_ADDDS'
                    config.T = 10
                    config.num_estimator_batches = 1
                    config.num_estimator_samples = 10000
                    config.sampling_eps = 5e-3
                elif method == 'ZOD-MC':
                    distribution.keep_minimizer = True
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection'
                    config.optimizer_reg = 'False'
                    config.sampling_method = 'ei'
                    config.T = 10
                    config.num_estimator_batches = 1
                    config.num_estimator_samples = 10000
                    config.sampling_eps = 5e-3
                elif method == 'RDMC':
                    distribution.keep_minimizer = False
                    config.score_method = 'p0t'
                    config.p0t_method = 'ula'
                    config.sampling_method = 'ei'
                    config.T = 2
                    config.num_estimator_batches = 1
                    config.num_estimator_samples = 1000
                    config.num_sampler_iterations = 100
                    config.ula_step_size = 0.01
                    config.sampling_eps = 5e-2
                elif method == 'RSDMC':
                    config.score_method = 'recursive'
                    config.sampling_method = 'ei'
                    config.T = 2
                    config.num_estimator_batches = 1
                    config.num_recursive_steps = 3
                    config.num_estimator_samples = 10
                    config.num_sampler_iterations = 3
                    config.ula_step_size = 0.01
                    config.sampling_eps = 5e-2
                elif method == 'SLIPS':
                    config.score_method = 'p0t'
                    config.p0t_method = 'mala'
                    config.sampling_method = 'ei'
                    config.num_estimator_batches = 1
                    config.num_estimator_samples = 100
                    config.slips_mala_steps = 500
                    config.sampling_eps = getattr(config, 'sampling_eps_slips', config.sampling_eps_rdmc)

                samples_all[k][i] = sample.sample(config,distribution)
                mmd_stats[k][i] = mmd.get_mmd_squared(samples_all[k][i],samples_all[0][i]).detach().item()
                w2_stats[k][i] = utils.metrics.get_w2(samples_all[k][i],samples_all[0][i]).detach().item()

                k+=1

            for method in config.baselines:
                print(method, r)
                in_cond = torch.randn_like(samples_all[0][i])
                if method == 'langevin':
                    distribution.keep_minimizer = False
                    ula_step_size = 0.01
                    num_steps_lang = 50000
                    samples_all[k][i] = samplers.ula.get_ula_samples(in_cond,
                                                                    distribution.grad_log_prob,
                                                                    ula_step_size,num_steps_lang,display_pbar=False)
                elif method == 'proximal':
                    samples_all[k][i] = samplers.proximal_sampler.get_samples(in_cond,
                                                                            distribution,
                                                                            config.proximal_M,
                                                                            config.proximal_num_iters,
                                                                            1,device
                                                                            ).squeeze(1)
                elif method == 'parallel':
                    num_chains = config.num_chains_parallel
                    num_iters = 10000
                    betas = torch.linspace(.2,1.,num_chains, dtype=torch.float32,device=device)
                    samples_all[k][i], _ = samplers.parallel_tempering.parallel_tempering(distribution,
                                                                                          in_cond, betas, num_iters,
                                                                                          config.langevin_step_size,
                                                                                          device)
                elif method == 'smc':
                    smc_beta_proposal_scale = getattr(config, 'smc_beta_proposal', 0.2)
                    smc_mcmc_steps = 100
                    samples_all[k][i] = smc_sampler.smc_sampler_for_zodmc(
                        target_dist=distribution,
                        dim=config.dimension,
                        n_particles=tot_samples,
                        n_mcmc_steps_per_particle=smc_mcmc_steps,
                        beta_smc=smc_beta_proposal_scale,
                        device=device,
                        noise_std_per_epoch=config.added_noise_std_for_reg
                    )
                mmd_stats[k][i] = mmd.get_mmd_squared(samples_all[k][i],samples_all[0][i]).detach().item()
                w2_stats[k][i] = utils.metrics.get_w2(samples_all[k][i],samples_all[0][i]).detach().item()
                mass_center[k][i] = get_mass_center(config,samples_all[k][i],r)
                k+=1
            xlim = [-4, r + 8]
            ylim = [-4, r + 8]
            fig = utils.plots.plot_all_samples(samples_all[:,i,:,:],
                                            method_names,
                                            xlim,ylim,distribution.log_prob)
            fig.savefig(os.path.join(folder,f'radius_{r}.png'), bbox_inches='tight')
            plt.close(fig)
    else:
        samples_all = torch.load(config.samples_ckpt).to(device=device).to(dtype=torch.float32)
        method_names = np.load(os.path.join(folder,'method_names.npy'))
        for i, r in enumerate(radiuses):
            for k, method in enumerate(method_names):
                if method == 'Ground Truth':
                    k-=1
                    continue
                distribution = get_gmm_radius(config,r,device)

                mmd_stats[k][i] = mmd.get_mmd_squared(samples_all[k][i],samples_all[0][i]).detach().item()
                w2_stats[k][i] = utils.metrics.get_w2(samples_all[k][i],samples_all[0][i]).detach().item()
                mass_center[k][i] = get_mass_center(config,samples_all[k][i],r)
                print(f'{method} {r} {torch.sum((samples_all[k][i][:,0] < 30))} {torch.sum((samples_all[k][i][:,1] < 30))}')
                xlim = [-4, r + 8]
                ylim = [-4, r + 8]
            fig = utils.plots.plot_all_samples(samples_all[:,i,:,:],
                                            method_names,
                                            xlim,ylim,distribution.log_prob)
            fig.savefig(os.path.join(folder,f'radius_{r}.png'), bbox_inches='tight')
            plt.close(fig)
    save_file = os.path.join(folder,f'samples_{config.density}.pt')
    np.save(os.path.join(folder,'method_names.npy'), np.array(method_names))
    torch.save(samples_all, save_file)
    plt.rcParams.update({'font.size': 14})

    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(18,6))
    ls=['--','-.',':']
    markers=['p','*','s','d','h']

    for i,method in enumerate(method_names):
        method_label = method[0].upper() + method[1:]
        if method == 'Ground Truth':
            continue
        print(method)
        ax1.plot(radiuses,mmd_stats[i,:radiuses.shape[0]],label=method_label,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
        ax2.plot(radiuses,w2_stats[i, :radiuses.shape[0]],label=method_label,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
        ax3.plot(radiuses,mass_center[i,:radiuses.shape[0]],label=method_label,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
    ax1.set_xlabel('Radius')
    ax1.set_ylabel('MMD')
    ax1.legend(loc='upper left',bbox_to_anchor=(0.55,0.8))
    ax2.set_xlabel('Radius')
    ax2.set_ylabel('W2')
    ax2.legend(loc='upper left')

    ax3.set_yticks(np.arange(0.1, 1.1, 0.1))
    ax3.axhline(y=.1, label='True\nWeight',color='black',linestyle='dotted')
    ax3.set_xlabel('Radius')
    ax3.set_ylabel('Mass on Center Mode')
    ax3.legend(loc='upper left',bbox_to_anchor=(0.6,0.7))
    fig.savefig(os.path.join(folder,'radius_mmd_results.pdf'),bbox_inches='tight')


