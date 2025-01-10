import numpy as np
import torch
from torch import nn

from nflows.distributions.base import Distribution
from nflows.utils import torchutils

class ConditionalDiagonalStudentT(Distribution):
    """A diagonal multivariate Student-t whose parameters are functions of a context."""

    def __init__(self, shape, context_encoder=None, nu=3.0):
        """
        Args:
            shape: list, tuple, or torch.Size, the shape of the input variables.
            context_encoder: callable or None, encodes the context to the distribution parameters.
                If None, defaults to the identity function.
            nu: degrees of freedom for the Student-t distribution.
        """
        super().__init__()
        self._shape = torch.Size(shape)
        self.nu = nu  # degrees of freedom for Student-t
        if context_encoder is None:
            self._context_encoder = lambda x: x
        else:
            self._context_encoder = context_encoder

    def _compute_params(self, context):
        """Compute the means and log scales from the context."""
        if context is None:
            raise ValueError("Context can't be None.")
        
        params = self._context_encoder(context)
        if params.shape[-1] % 2 != 0:
            raise RuntimeError("The context encoder must return a tensor whose last dimension is even.")
        if params.shape[0] != context.shape[0]:
            raise RuntimeError("The batch dimension of the parameters is inconsistent with the input.")

        split = params.shape[-1] // 2
        means = params[..., :split].reshape(params.shape[0], *self._shape)
        log_scales = params[..., split:].reshape(params.shape[0], *self._shape)
        return means, log_scales

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                f"Expected input of shape {self._shape}, got {inputs.shape[1:]}"
            )

        # Compute parameters.
        means, log_scales = self._compute_params(context)
        assert means.shape == inputs.shape and log_scales.shape == inputs.shape

        # Create the Student-t distribution.
        dist = torch.distributions.StudentT(df=self.nu, loc=means, scale=torch.exp(log_scales))
        
        # Compute log prob.
        log_prob = dist.log_prob(inputs).sum(dim=-1)
        return log_prob

    def _sample(self, num_samples, context):
        # Compute parameters.
        means, log_scales = self._compute_params(context)
        scales = torch.exp(log_scales)

        #means = torchutils.repeat_rows(means, num_samples)
        #scales = torchutils.repeat_rows(scales, num_samples)

        # Generate Student-t samples.
        context_size = context.shape[0]
        dist = torch.distributions.StudentT(df=self.nu, loc=means, scale=scales)
        samples = dist.sample([context_size * num_samples]).squeeze(1)

        return torchutils.split_leading_dim(samples, [context_size, num_samples])
        

    def _mean(self, context):
        """The mean for Student-t is defined as the location parameter (if ν > 1)."""
        means, _ = self._compute_params(context)
        return means if self.nu > 1 else float('nan')  # Mean exists only for ν > 1.



class ConditionalDiagonalNormal(Distribution):
    """A diagonal multivariate Normal whose parameters are functions of a context."""

    def __init__(self, shape, context_encoder=None):
        """Constructor.

        Args:
            shape: list, tuple or torch.Size, the shape of the input variables.
            context_encoder: callable or None, encodes the context to the distribution parameters.
                If None, defaults to the identity function.
        """
        super().__init__()
        self._shape = torch.Size(shape)
        if context_encoder is None:
            self._context_encoder = lambda x: x
        else:
            self._context_encoder = context_encoder
        self.register_buffer("_log_z",
                             torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
                                          dtype=torch.float64),
                             persistent=False)

    def _compute_params(self, context):
        """Compute the means and log stds form the context."""
        if context is None:
            raise ValueError("Context can't be None.")

        params = self._context_encoder(context)
        if params.shape[-1] % 2 != 0:
            raise RuntimeError(
                "The context encoder must return a tensor whose last dimension is even."
            )
        if params.shape[0] != context.shape[0]:
            raise RuntimeError(
                "The batch dimension of the parameters is inconsistent with the input."
            )

        split = params.shape[-1] // 2
        means = params[..., :split].reshape(params.shape[0], *self._shape)
        log_stds = params[..., split:].reshape(params.shape[0], *self._shape)
        return means, log_stds

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )

        # Compute parameters.
        means, log_stds = self._compute_params(context)
        assert means.shape == inputs.shape and log_stds.shape == inputs.shape

        # Compute log prob.
        norm_inputs = (inputs - means) * torch.exp(-log_stds)
        log_prob = -0.5 * torchutils.sum_except_batch(
            norm_inputs ** 2, num_batch_dims=1
        )
        log_prob -= torchutils.sum_except_batch(log_stds, num_batch_dims=1)
        log_prob -= self._log_z
        return log_prob

    def _sample(self, num_samples, context):
        # Compute parameters.
        means, log_stds = self._compute_params(context)
        stds = torch.exp(log_stds)
        means = torchutils.repeat_rows(means, num_samples)
        stds = torchutils.repeat_rows(stds, num_samples)

        # Generate samples.
        context_size = context.shape[0]
        noise = torch.randn(context_size * num_samples, *
                            self._shape, device=means.device)
        samples = means + stds * noise
        return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        means, _ = self._compute_params(context)
        return means
