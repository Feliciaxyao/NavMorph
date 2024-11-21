import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    def __init__(self, norm, channel_dim=-1):
        super().__init__()
        self.norm = norm
        self.channel_dim = channel_dim

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        loss = self.loss_fn(prediction.float(), target,float(), reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=self.channel_dim, keepdims=True)
        return loss.mean()


class SpatialRegressionLoss(nn.Module):
    def __init__(self, norm, ignore_index=255):
        super(SpatialRegressionLoss, self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        assert len(prediction.shape) == 5, 'Must be a 5D tensor'
        # ignore_index is the same across all channels
        mask = target[:, :, :1] != self.ignore_index
        if mask.sum() == 0:
            return prediction.new_zeros(1)[0].float()

        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=-3, keepdims=True)

        return loss[mask].mean()


class ProbabilisticLoss(nn.Module):
    """ Given a prior distribution and a posterior distribution, this module computes KL(posterior, prior)"""
    def __init__(self, remove_first_timestamp=True):
        super().__init__()
        self.remove_first_timestamp = remove_first_timestamp

    def forward(self, prior_mu, prior_sigma, posterior_mu, posterior_sigma):
        posterior_var = posterior_sigma[:, 1:] ** 2
        prior_var = prior_sigma[:, 1:] ** 2

        posterior_log_sigma = torch.log(posterior_sigma[:, 1:])
        prior_log_sigma = torch.log(prior_sigma[:, 1:])

        kl_div = (
                prior_log_sigma - posterior_log_sigma - 0.5
                + (posterior_var + (posterior_mu[:, 1:] - prior_mu[:, 1:]) ** 2) / (2 * prior_var)
        )
        first_kl = - posterior_log_sigma[:, :1] - 0.5 + (posterior_var[:, :1] + posterior_mu[:, :1] ** 2) / 2
        kl_div = torch.cat([first_kl, kl_div], dim=1)

        # Sum across channel dimension
        # Average across batch dimension, keep time dimension for monitoring
        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))
        return kl_loss


class KLLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.loss = ProbabilisticLoss(remove_first_timestamp=True)

    def forward(self, prior, posterior):
        prior_mu, prior_sigma = prior['mu'], prior['sigma']
        posterior_mu, posterior_sigma = posterior['mu'], posterior['sigma']
        prior_mu = prior_mu.float()
        prior_sigma = prior_sigma.float()
        posterior_mu = posterior_mu.float()
        posterior_sigma = posterior_sigma.float()

        prior_loss = self.loss(prior_mu, prior_sigma, posterior_mu.detach(), posterior_sigma.detach())
        posterior_loss = self.loss(prior_mu.detach(), prior_sigma.detach(), posterior_mu, posterior_sigma)

        return self.alpha * prior_loss + (1 - self.alpha) * posterior_loss