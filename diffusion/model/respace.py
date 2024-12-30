# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import math

import numpy as np
import torch as th

from ..model import gaussian_diffusion as gd
from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = th.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = th.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = th.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (th.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = th.rand(size=(batch_size,), device="cpu")
    return u


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        flow_shift = kwargs.pop("flow_shift")
        diffusion_steps = kwargs.pop("diffusion_steps")
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        if kwargs.get("model_mean_type", False) != gd.ModelMeanType.VELOCITY:
            new_betas = []
            for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
                if i in self.use_timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
                    self.timestep_map.append(i)
            kwargs["betas"] = np.array(new_betas)
            super().__init__(**kwargs)
        else:
            new_sigmas = flow_shift * base_diffusion.sigmas / (1 + (flow_shift - 1) * base_diffusion.sigmas)
            self.timestep_map = new_sigmas * diffusion_steps
            # self.timestep_map = list(self.use_timesteps)
            kwargs["sigmas"] = np.array(new_sigmas)
            super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def training_losses_diffusers(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        return super().training_losses_diffusers(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.use_timesteps)

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, use_timesteps):
        self.model = model
        self.timestep_map = timestep_map
        self.use_timesteps = use_timesteps
        
        # Handle DistributedDataParallel wrapper
        if hasattr(model, 'module'):
            base_model = model.module
        else:
            base_model = model
            
        
        # Try to find VAE in the base model
        self.vae = None
        if hasattr(base_model, 'vae'):
            self.vae = base_model.vae
        elif hasattr(base_model, 'sana') and hasattr(base_model.sana, 'vae'):
            self.vae = base_model.sana.vae
            

    def __call__(self, x, timestep, **kwargs):
        """
        :param x: An [N x C x ...] Tensor of inputs.
        :param timestep: A 1-D batch of timesteps.
        :return: An [N x C x ...] Tensor of outputs.
        """
        # Map from original diffusion steps to compressed diffusion steps
        if isinstance(self.timestep_map, (np.ndarray, list)):
            timestep = np.array([self.timestep_map[t] for t in timestep.cpu()])
            timestep = th.from_numpy(timestep).to(x.device)
        
        return self.model(x, timestep, **kwargs)
