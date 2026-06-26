import configargparse
import mmd_loss_comparisons
import radius_increase_experiments
import dimension_increase_experiments

def parse_arguments():
    p = configargparse.ArgParser(description='Arguments for nonconvex sampling')

    p.add('-c','--config', is_config_file=True)

    p.add_argument('--mode', choices=['eval_mmd','radius','dimension'])
    p.add_argument('--score_method', choices=['p0t','recursive'],default='p0t')
    p.add_argument('--p0t_method', choices=['rejection','ula','rejection_hessian'],default='rejection')
    p.add_argument('--dimension', type=int)

    p.add_argument('--load_from_ckpt',action='store_true')
    p.add_argument('--samples_ckpt',type=str)
    p.add_argument('--save_folder',type=str)

    p.add_argument('--eval_mmd', action='store_true',default=False)
    p.add_argument('--methods_to_run',action='append', default=[])
    p.add_argument('--num_samples_for_rdmc',type=int)
    p.add_argument('--sampling_eps_rdmc', type=float)
    p.add_argument('--sampling_eps_rejec', type=float)

    p.add_argument('--min_num_iters_rdmc',type=int)
    p.add_argument('--max_num_iters_rdmc',type=int)
    p.add_argument('--iters_rdmc_step',type=int)

    p.add_argument('--baselines',action='append', default=[])
    p.add_argument('--langevin_step_size',type=float)

    p.add_argument('--proximal_M',type=float)
    p.add_argument('--proximal_num_iters',type=int)
    p.add_argument('--num_chains_parallel',type=int,default=6)

    p.add_argument('--max_iters_optimization',type=int, default=50)
    p.add_argument('--num_sampler_iterations', type=int)
    p.add_argument('--ula_step_size',type=float)
    p.add_argument('--rdmc_initial_condition',choices=['normal','delta'],default='normal')
    p.add_argument('--num_estimator_batches', type=int, default=1)
    p.add_argument('--num_estimator_samples', type=int, default=10000)
    p.add_argument('--eps_stable',type=float, default=1e-9)
    p.add_argument('--num_recursive_steps',type=int, default=6)

    p.add_argument('--added_noise_std_for_reg', type=float, default=0.01)
    p.add_argument('--optimizer_reg_lambda', type=float, default=0.01)
    p.add_argument('--optimizer_reg', type=str,)
    p.add_argument('--perturbation_mode', choices=['gaussian', 'fgsm', 'pgd', 'gcg'])

    p.add_argument('--pgd_eps', type=float, default=0.5, help='Epsilon for PGD attack')
    p.add_argument('--pgd_alpha', type=float, default=0.01, help='Step size for PGD attack')
    p.add_argument('--pgd_steps', type=int, default=10, help='Number of steps for PGD attack')
    p.add_argument('--fgsm_eps', type=float, default=0.01, help='Epsilon for FGSM attack')

    p.add_argument('--gcg_eps', type=float, default=0.5, help='Epsilon for GCG attack')
    p.add_argument('--gcg_alpha', type=float, default=0.01, help='Step size for GCG attack')
    p.add_argument('--gcg_steps', type=int, default=10, help='Number of steps for GCG attack')
    p.add_argument('--gcg_k', type=int, default=1, help='Number of coordinates to perturb per step in GCG')

    p.add_argument('--smc_perturbation_mode', choices=['gaussian', 'fgsm', 'pgd', 'none', 'gcg'], default='none',
                   help='Perturbation mode for SMC baseline')
    p.add_argument('--smc_pgd_steps', type=int, default=10, help='Number of PGD steps for SMC attack')
    p.add_argument('--smc_pgd_eps', type=float, default=0.03, help='Epsilon for PGD attack in SMC')
    p.add_argument('--smc_pgd_alpha', type=float, default=0.01, help='Alpha (step size) for PGD attack in SMC')
    p.add_argument('--smc_fgsm_eps', type=float, default=0.01, help='Epsilon for FGSM attack in SMC')

    p.add_argument('--sde_type', choices=['vp'], default='vp')
    p.add_argument('--multiplier', default=0, type=float)
    p.add_argument('--bias', default=2., type=float)


    p.add_argument('--sampling_method', choices=['ei','em','ei_ADDDS'])
    p.add_argument('--num_batches', type=int)
    p.add_argument('--sampling_batch_size',type=int)
    p.add_argument('--T', type=float)
    p.add_argument('--sampling_eps', type=float)
    p.add_argument('--disc_steps',type=int)
    p.add_argument('--ula_steps',type=int,default=0)


    p.add_argument('--density',choices=['gmm','mueller','lmm'])
    p.add_argument('--density_parameters_path',type=str)
    p.add_argument('--discontinuity',action='store_true',default=False)


    p.add_argument('--use_score_clipping', action='store_true', default=False,
                   help='Enable Gradient/Score Clipping defense.')
    p.add_argument('--score_clip_norm', type=float, default=10.0,
                   help='The maximum norm for score clipping.')

    p.add_argument('--use_pc_sampler', action='store_true', default=False,
                   help='Enable Predictor-Corrector (Langevin correction) as a robust integrator.')
    p.add_argument('--pc_snr', type=float, default=0.16,
                   help='Signal-to-noise ratio for the corrector step.')


    return p.parse_args()

def main(config):
    if config.mode == 'eval_mmd':
        mmd_loss_comparisons.eval(config)
    elif config.mode == 'radius':
        radius_increase_experiments.eval(config)
    elif config.mode == 'dimension':
        dimension_increase_experiments.eval(config)
    else:
        print("Mode hasn't been implemented")



if __name__ == '__main__':
    config = parse_arguments()
    print(config)
    main(config)