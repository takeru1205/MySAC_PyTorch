import math
import torch


def calculate_log_pi(log_stds, noises, actions):
    """Return probability density of stochastic action"""
    gaussian_log_probs = \
        (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)
    log_pis = gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
    return log_pis


def reparameterize(means, log_stds):
    """Reparameterization Trick"""
    stds = log_stds.exp()  # sigma(hyouzyunhensa)
    noises = torch.randn_like(means)  # epsilon
    us = means + noises * stds  # u = mu + epsilon + sigma
    actions = torch.tanh(us)  # Equation 8

    log_pis = calculate_log_pi(log_stds, noises, actions)  # log( \pi( a_t | s_t ))
    return actions, log_pis
