import yaml
import torch

from utils.densities import MultivariateGaussian, MixtureDistribution

def get_gmm_density_at_t(config, sde, t, device):
    def to_tensor_type(x):
        return torch.tensor(x,device=device, dtype=torch.float32)
    params = yaml.safe_load(open(config.density_parameters_path))

    c = to_tensor_type(params['coeffs'])
    means = to_tensor_type(params['means'])
    variances = to_tensor_type(params['variances'])
    dist = get_gmm_density_at_t_no_config(sde, t, c, means, variances)
    return dist.log_prob, dist.gradient

def get_gmm_density_at_t_no_config(sde, t, weights, means, variances):
    scale = sde.scaling(t)
    if torch.is_tensor(scale) and scale.ndim == 0:
        pass
    elif torch.is_tensor(scale) and scale.ndim > 0:
        scale = scale.view(-1, 1)

    mean_t = means * scale
    eye = torch.eye(means.shape[-1], device=variances.device)
    scale_sq = scale ** 2
    if variances.ndim == 3:
        var_t = variances * scale_sq + (1 - scale_sq) * eye.unsqueeze(0)
    elif variances.ndim == 2:
        var_t = variances * scale_sq + (1 - scale_sq) * eye

    gaussians = [MultivariateGaussian(mean_t[i], var_t[i]) for i in range(len(weights))]
    dist = MixtureDistribution(weights, gaussians)

    return dist