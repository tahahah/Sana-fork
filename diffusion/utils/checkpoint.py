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

import os
import random
import re
import huggingface_hub
import numpy as np
import torch

from diffusion.utils.logger import get_root_logger
from tools.download import find_model
from dotenv import load_dotenv
from pathlib import Path

# Go up three levels: utils -> diffusion -> Sana-fork (project root)
load_dotenv(Path(__file__).parent.parent.parent / '.env')


def save_checkpoint(
    work_dir,
    epoch,
    model,
    model_ema=None,
    optimizer=None,
    lr_scheduler=None,
    generator=torch.Generator(device="cpu").manual_seed(42),
    keep_last=False,
    step=None,
    add_symlink=False,
    upload_to_hub=True,
):
    logger = get_root_logger()
    try:
        os.makedirs(work_dir, exist_ok=True)
        state_dict = dict(state_dict=model.state_dict())
        if model_ema is not None:
            state_dict["state_dict_ema"] = model_ema.state_dict()
        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None:
            state_dict["scheduler"] = lr_scheduler.state_dict()
        if epoch is not None:
            state_dict["epoch"] = epoch
            file_path = os.path.join(work_dir, f"epoch_{epoch}.pth")
            if step is not None:
                file_path = file_path.split(".pth")[0] + f"_step_{step}.pth"

        rng_state = {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "generator": generator.get_state(),
        }
        state_dict["rng_state"] = rng_state

        torch.save(state_dict, file_path)
        logger.info(f"Saved checkpoint of epoch {epoch} to {file_path}.")

        if keep_last:
            for i in range(epoch):
                previous_ckgt = file_path.format(i)
                if os.path.exists(previous_ckgt):
                    os.remove(previous_ckgt)
        if add_symlink:
            link_path = os.path.join(os.path.dirname(file_path), "latest.pth")
            if os.path.exists(link_path) or os.path.islink(link_path):
                os.remove(link_path)
            os.symlink(os.path.abspath(file_path), link_path)

        # Upload to HuggingFace if requested
        if upload_to_hub and "HF_TOKEN" in os.environ:
            try:
                huggingface_hub.login(token=os.environ["HF_TOKEN"])
                repo_id = "Tahahah/pacman-sana-3.2m"
                
                try:
                    huggingface_hub.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=f"checkpoints/{os.path.basename(file_path)}",
                        repo_id=repo_id,
                        repo_type="model"
                    )
                    logger.info(f"Uploaded checkpoint to HuggingFace: {repo_id}")
                except huggingface_hub.utils.RepositoryNotFoundError:
                    huggingface_hub.create_repo(repo_id, repo_type="model")
                    huggingface_hub.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=f"checkpoints/{os.path.basename(file_path)}",
                        repo_id=repo_id,
                        repo_type="model"
                    )
                    logger.info(f"Created repo and uploaded checkpoint to HuggingFace: {repo_id}")
            except Exception as e:
                logger.warning(f"Failed to upload checkpoint to HuggingFace: {e}")

        return file_path
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return None


def load_checkpoint(
    checkpoint,
    model,
    model_ema=None,
    optimizer=None,
    lr_scheduler=None,
    load_ema=False,
    resume_optimizer=True,
    resume_lr_scheduler=True,
    null_embed_path=None,
):
    logger = get_root_logger()
    try:
        assert isinstance(checkpoint, str)
        ckpt_file = checkpoint
        checkpoint = find_model(ckpt_file)

        state_dict_keys = ["pos_embed", "base_model.pos_embed", "model.pos_embed"]
        for key in state_dict_keys:
            if key in checkpoint["state_dict"]:
                del checkpoint["state_dict"][key]
                if "state_dict_ema" in checkpoint and key in checkpoint["state_dict_ema"]:
                    del checkpoint["state_dict_ema"][key]
                break

        if load_ema:
            state_dict = checkpoint["state_dict_ema"]
        else:
            state_dict = checkpoint.get("state_dict", checkpoint)

        # Try to load null_embed from local path or HuggingFace
        try:
            null_embed = torch.load(null_embed_path, map_location="cpu")
        except FileNotFoundError:
            try:
                if "HF_TOKEN" in os.environ:
                    huggingface_hub.login(token=os.environ["HF_TOKEN"])
                    null_embed_file = huggingface_hub.hf_hub_download(
                        repo_id="Tahahah/pacman-sana-3.2m",
                        filename="pretrained_models/null_embed_diffusers_dc-ae_16.pth",
                        repo_type="model"
                    )
                    null_embed = torch.load(null_embed_file, map_location="cpu")
                    logger.info("Loaded null_embed from HuggingFace")
                else:
                    raise FileNotFoundError("HF_TOKEN not found in environment")
            except Exception as e:
                logger.warning(f"Could not load null_embed from HuggingFace: {e}")
                logger.warning("Proceeding without null_embed")
                null_embed = None

        if null_embed is not None:
            state_dict["y_embedder.y_embedding"] = null_embed["uncond_prompt_embeds"][0]
        
        rng_state = checkpoint.get("rng_state", None)

        missing, unexpect = model.load_state_dict(state_dict, strict=False)
        if model_ema is not None:
            model_ema.load_state_dict(checkpoint["state_dict_ema"], strict=False)
        if optimizer is not None and resume_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None and resume_lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])

        epoch = checkpoint.get("epoch", 0)
        logger.info(f"Successfully loaded checkpoint from {ckpt_file}")
        return epoch, missing, unexpect, rng_state

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return 0, [], [], None
