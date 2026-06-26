
import torch
from tqdm import tqdm


def get_mala_samples(xk, log_prob_target, grad_log_prob_target, h, num_iters, display_pbar=True):

    yk = xk.detach().clone()
    accepted_count = 0

    for _ in tqdm(range(num_iters), leave=False, disable=display_pbar):
        current_yk = yk.detach().clone()
        grad_current_yk = grad_log_prob_target(current_yk)
        proposal_yk = current_yk + grad_current_yk * h + (2 * h) ** .5 * torch.randn_like(current_yk)
        log_prob_current = log_prob_target(current_yk)
        log_prob_proposal = log_prob_target(proposal_yk)
        grad_proposal_yk = grad_log_prob_target(proposal_yk)
        log_q_current_given_proposal = -torch.sum((current_yk - proposal_yk - grad_proposal_yk * h) ** 2, dim=-1) / (4 * h)
        log_q_proposal_given_current = -torch.sum((proposal_yk - current_yk - grad_current_yk * h) ** 2, dim=-1) / (4 * h)
        if log_prob_current.ndim == 1:
            log_prob_current = log_prob_current.unsqueeze(-1)
        if log_prob_proposal.ndim == 1:
            log_prob_proposal = log_prob_proposal.unsqueeze(-1)
        if log_q_current_given_proposal.ndim == 1:
            log_q_current_given_proposal = log_q_current_given_proposal.unsqueeze(-1)
        if log_q_proposal_given_current.ndim == 1:
            log_q_proposal_given_current = log_q_proposal_given_current.unsqueeze(-1)

        log_acceptance_ratio = (log_prob_proposal + log_q_current_given_proposal) -\
                               (log_prob_current + log_q_proposal_given_current)

        acceptance_ratio = torch.exp(torch.clamp(log_acceptance_ratio, max=0))

        u = torch.rand_like(acceptance_ratio)
        accept_mask = (u < acceptance_ratio)

        yk[accept_mask.squeeze(-1)] = proposal_yk[accept_mask.squeeze(-1)]
        accepted_count += accept_mask.sum().item()


    return yk