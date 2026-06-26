import os
import torch
import numpy as np
import utils.plots
import utils.densities
import utils.metrics
import sample
import matplotlib.pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_gmm_dimension(D, num_modes,device):
    setup_seed(D)
    c = torch.ones(num_modes, device=device)/D
    noise = torch.rand((num_modes,D), device=device)
    means = noise/(torch.sum(noise**2,dim=-1,keepdim=True)**.5) * 6
    variances = torch.eye(D,device=device).unsqueeze(0).expand((num_modes,D,D))
    variances = variances * (torch.rand((num_modes,1,1),device=device) + 0.3 )
    gaussians = [utils.densities.MultivariateGaussian(means[i],variances[i]) for i in range(c.shape[0])]
    return utils.densities.MixtureDistribution(c,gaussians)

def compute_statistic(distribution : utils.densities.MixtureDistribution, samples):
    f = 0
    for dist in distribution.distributions:
        f += torch.mean(torch.sum((samples-dist.mean)**2,dim=-1),dim=0)
    return f

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

    tot_samples = config.num_batches * config.sampling_batch_size
    num_methods, method_names = get_method_names(config)
    dimensions = np.array([50])
    num_dims = len(dimensions)
    stats = np.zeros([num_methods, num_dims],dtype='double')
    w2_stats = np.zeros([num_methods, num_dims],dtype='double')

    num_modes = 5

    folder = os.path.dirname(config.save_folder)
    os.makedirs(folder, exist_ok=True)

    if not config.load_from_ckpt:
        for i, d in enumerate(dimensions):
            config.dimension = d
            distribution = get_gmm_dimension(d,num_modes,device)


            true_samples = distribution.sample(tot_samples)
            stats[0][i] = compute_statistic(distribution, true_samples)
            k = 1
            for method in config.methods_to_run:
                print(f'{method} {d}')
                if method == 'ADDDS':
                    distribution.keep_minimizer = True
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection'
                    config.optimizer_reg = 'True'
                    config.sampling_method = 'ei_ADDDS'
                    config.T = 2
                    config.num_estimator_batches = 10 * d
                    config.num_estimator_samples = 10000
                    config.sampling_eps = 5e-3
                elif method == 'ZOD-MC':
                    distribution.keep_minimizer = True
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection'
                    config.sampling_method = 'ei'
                    config.T = 2
                    config.num_estimator_batches = 10 * d
                    config.num_estimator_samples = 10000
                    config.sampling_eps = 5e-3
                elif method == 'Hessian':
                    distribution.keep_minimizer = True
                    config.score_method = 'p0t'
                    config.p0t_method = 'rejection_hessian'
                    config.sampling_method = 'ei'
                    config.T = 2
                    config.num_estimator_batches = 10 * d
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
                    config.ula_step_size = 0.1
                    config.sampling_eps = 5e-2
                elif method == 'RSDMC':
                    config.score_method = 'recursive'
                    config.sampling_method = 'ei'
                    config.T = 2
                    config.num_estimator_batches = 1
                    config.num_recursive_steps = 3
                    config.num_estimator_samples = 10
                    config.num_sampler_iterations = 5
                    config.ula_step_size = 0.1
                    config.sampling_eps = 5e-2
                elif method == 'SLIPS':
                    config.score_method = 'p0t'
                    config.p0t_method = 'mala'
                    config.sampling_method = 'ei'
                    config.num_estimator_batches = 1
                    config.num_estimator_samples = 100
                    config.slips_mala_steps = 100
                    config.sampling_eps = getattr(config, 'sampling_eps_slips', config.sampling_eps_rdmc)

                generated_samples = sample.sample(config,distribution)
                stats[k][i] = compute_statistic(distribution, generated_samples)
                w2_stats[k][i] = utils.metrics.get_w2(generated_samples,true_samples).detach().item()
                k+=1
    else:
        stats = torch.load(os.path.join(folder,'log_z.pt'))
        w2_stats = torch.load(os.path.join(folder,'w2.pt'))

        method_names = np.load(os.path.join(folder,'method_names.npy'))

    torch.save(stats,os.path.join(folder,'log_z.pt'))
    torch.save(w2_stats,os.path.join(folder,'w2.pt'))

    np.save(os.path.join(folder,'method_names.npy'), np.array(method_names))
    plt.rcParams.update({
        'font.size': 20,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
    ls=['--','-.',':']
    markers=['p','*','s','d','h']
    print(stats)
    for i,method in enumerate(method_names):
        method_label = method[0].upper() + method[1:]
        print(method)
        ax1.plot(dimensions,np.abs(stats[i]-stats[0]),label=method_label,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
        ax2.plot(dimensions,w2_stats[i],label=method_label,linestyle=ls[i%3],marker=markers[i%5],markersize=7)
    ax1.set_xlabel('Dimension')

    ax1.set_ylabel(r'Error in estimation of $\mathbb{E}[f(x)]$')
    ax1.set_ylim(0,800)
    ax1.legend(loc='upper left')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('W2')
    ax2.legend(loc='upper left')
    fig.savefig(os.path.join(folder,'dimension_mmd_results.pdf'),bbox_inches='tight')


