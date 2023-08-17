# --- Libraries
import torch
import torch.nn.functional as F  # Noqa

from helpers import get_device
# ---


def relu_evidence(y):
    return F.relu(y)  # ReLU(logits)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def get_annealing_coefficient(epoch_num, annealing_step):
    return torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )


def get_evidence_alpha(output):
    evidence = relu_evidence(output)  # Predicted evidence, f(x_i, theta) = ReLU(logits)
    alpha = evidence + 1  # Dirichlet parameters, alpha_i = f(x_i, theta) + 1

    return evidence, alpha


def dirichlet_strength(alpha):
    # Dirichlet strength (S), S = sum(alpha_i)
    return torch.sum(alpha, dim=1, keepdim=True)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = dirichlet_strength(alpha)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )

    return first_term + second_term  # kl

def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()

    # --- Move input and label tensors to the specified device
    y = y.to(device)
    alpha = alpha.to(device)
    # ---

    S = dirichlet_strength(alpha)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )

    return log_likelihood_err + log_likelihood_var  # log_likelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()

    # --- Move input and label tensors to the specified device
    y = y.to(device)
    alpha = alpha.to(device)
    # ---

    log_likelihood = log_likelihood_loss(y, alpha, device=device)

    annealing_coeff = get_annealing_coefficient(epoch_num, annealing_step)

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coeff * kl_divergence(kl_alpha, num_classes, device=device)
    return log_likelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = dirichlet_strength(alpha)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coeff = get_annealing_coefficient(epoch_num, annealing_step)

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coeff * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()

    _, alpha = get_evidence_alpha(output)

    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()

    _, alpha = get_evidence_alpha(output)

    return torch.mean(
        edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step, device)
    )


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()

    _, alpha = get_evidence_alpha(output)

    return torch.mean(
        edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    )
