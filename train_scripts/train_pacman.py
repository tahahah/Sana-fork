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

import datetime
import getpass
import hashlib
import json
import os
import os.path as osp
import random
import time
import types
import warnings
from dataclasses import asdict
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
import itertools
import numpy as np
import pyrallis
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from PIL import Image
from termcolor import colored

warnings.filterwarnings("ignore")  # ignore warning


from diffusion import DPMS, FlowEuler, Scheduler
from diffusion.data.builder import build_dataloader, build_dataset
from diffusion.data.wids import DistributedRangedSampler
from diffusion.model.builder import build_model, get_vae, vae_decode, vae_encode
from diffusion.model.respace import compute_density_for_timestep_sampling
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion.utils.config import SanaConfig
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.utils.dist_utils import clip_grad_norm_, flush, get_world_size
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import DebugUnderflowOverflow, init_random_seed, read_config, set_random_seed
from diffusion.utils.optimizer import auto_scale_lr, build_optimizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "SanaBlock"


@torch.inference_mode()
def log_validation(accelerator, config, model, logger, step, device, vae=None, init_noise=None):
    print("\n=== Starting Validation ===")
    torch.cuda.empty_cache()
    global val_dataset, val_dataloader, val_iterator
    
    # Initialize validation dataset and dataloader if not already done
    if val_dataset is None:
        val_dataset = build_dataset(asdict(config.data), resolution=image_size, aspect_ratio_type=config.model.aspect_ratio_type, vae_downsample_rate=config.vae.vae_downsample_rate, vae=None)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.train.train_batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
            pin_memory=True
        )
        val_iterator = iter(val_dataloader)
    
    batch_size = config.train.train_batch_size

    # Get a random batch using our sampling function
    try:
        batch = random_sample_from_iterable(val_iterator, upper_limit=100)
    except StopIteration:
        # If we've exhausted the iterator, create a new one
        val_iterator = iter(val_dataloader)
        batch = next(val_iterator)
    
    vis_sampler = config.scheduler.vis_sampler
    model = accelerator.unwrap_model(model).eval()
    hw = torch.tensor([[config.model.image_size, config.model.image_size]], dtype=torch.float, device=device).repeat(1, 1)
    ar = torch.tensor([[1.0]], device=device).repeat(1, 1)
    
    # Create null action tensors for classifier-free guidance
    seq_len = config.data.sequence_length
    null_action = torch.zeros(batch_size, 1, seq_len-1, 5, device=device)  # 5 is number of actions
    null_action_mask = torch.ones(batch_size, seq_len-1, device=device)

    null_action = null_action.unsqueeze(1)
    null_action_mask = null_action_mask.unsqueeze(1)
    
    print(f"Created tensors:")
    print(f"- hw shape: {hw.shape}")
    print(f"- null_action shape: {null_action.shape}")
    
    # Create sampling noise:
    logger.info("Running validation... ")
    image_logs = []

    # Initialize VAE if not provided
    # if vae is None:
    #     vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, accelerator.device).to(torch.float16)
    #     # Set scaling factor from config
    #     if hasattr(vae, 'cfg') and vae.cfg.scaling_factor is None:
    #         vae.cfg.scaling_factor = config.vae.scale_factor

    def run_sampling(init_z=None, label_suffix="", vae=None, sampler="dpm-solver"):
        latents = []
        current_image_logs = []
        
        # Get a batch of validation samples from the dataset
        img = batch['img'].to(device)  # [B, C, H, W]
        obs = batch['obs'].to(device) # [B, S*C, H, W]
        actions = batch['y'].to(device)  # [B, S, A]
        action_masks = batch['y_mask'].to(device)  # [B, S]
        
        print(f"Debug shapes:")
        print(f"- img shape: {img.shape}")
        print(f"- obs shape: {obs.shape}")
        print(f"- actions shape before: {actions.shape}")
        print(f"- null_action shape: {null_action.shape}")
        
        # Reshape actions to match model expectations [B, 1, S, A]
        actions = actions.unsqueeze(1)  # Add the extra dimension
        
        print(f"- actions shape after: {actions.shape}")
        
        batch_size = img.shape[0]
        seq_len = img.shape[1] # seq_len*32
        
        # Generate initial noise if not provided
        z = init_z if init_z is not None else torch.randn_like(img)
        # encoded z = torch.randn([1, vae.cfg.latent_channels, img.shape[-2]//vae.cfg.latent_channels, img.shape[-1]//vae.cfg.latent_channels], device=device)
        print(f"Debug - initial z shape: {z.shape}")
        
        # Base model kwargs for the shape info and observations
        model_kwargs = dict(
            data_info={"img_hw": hw, "aspect_ratio": ar},
            mask=None,  # Use mask directly from dataset
            obs=obs,  # Pass observations to be concatenated with noise in the model
        )

        if sampler == "dpm-solver":
            dpm_solver = DPMS(
                model.sana.forward_with_dpmsolver,
                condition=actions,  # Use actions as condition
                uncondition=null_action,  # Use null action as uncondition
                cfg_scale=4.5,  # Same scale as original code
                model_kwargs=model_kwargs,
                model_type="flow",
                schedule="FLOW",
            )
            denoised = dpm_solver.sample(
                z,
                steps=40,
                order=2,
                skip_type="time_uniform_flow",
                method="multistep",
                flow_shift=config.scheduler.flow_shift,
            )
            print(f"Debug - denoised shape after dpm_solver: {denoised.shape if denoised is not None else None}")
        elif sampler == "flow_euler":
            flow_solver = FlowEuler(
                model, 
                condition=actions,  # Use actions as condition
                uncondition=null_action,  # Use null action as uncondition
                cfg_scale=4.5,  # Same scale as original code
                model_kwargs=model_kwargs
            )
            denoised = flow_solver.sample(
                z,
                steps=28
            )
            print(f"Debug - denoised shape after flow_euler: {denoised.shape if denoised is not None else None}")
        elif sampler == "flow_dpm-solver":
            dpm_solver = DPMS(
                model.forward_with_dpmsolver,
                condition=actions,  # Use actions as condition
                uncondition=null_action,  # Use null action as uncondition
                cfg_scale=4.5,  # Same scale as original code
                model_type="flow",
                model_kwargs=model_kwargs,
                schedule="FLOW",
            )
            denoised = dpm_solver.sample(
                z,
                steps=40,
                order=2,
                skip_type="time_uniform_flow",
                method="multistep",
                flow_shift=config.scheduler.flow_shift,
            )
            print(f"Debug - denoised shape after dpm_solver: {denoised.shape if denoised is not None else None}")
        else:
            raise ValueError(f"{sampler} not implemented")

        latents.append(denoised)
        
        torch.cuda.empty_cache()
        
        for latent in latents:
            print(f"Debug - latent: {latent.shape if latent is not None else None}")
            print(f"Debug - vae: {type(vae)}")
            if vae is not None:
                print(f"Debug - vae.cfg: {vae.cfg if hasattr(vae, 'cfg') else None}")
            
            latent = latent.to(torch.float16)
            # samples = vae_decode(config.vae.vae_type, vae, latent)
            samples = (
                torch.clamp(127.5 * latent + 128.0, 0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()[0]
            )
            print(f"Debug - samples shape: {samples.shape}")
            image = Image.fromarray(samples)
            
            # Convert actions to readable format for logging
            action_names = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'NO_ACTION']
            action_seq = []
            for i in range(seq_len):
                action_idx = actions[0, 0, 0, i].argmax().item()
                action_seq.append(action_names[action_idx])
            action_str = ' -> '.join(action_seq)
            
            current_image_logs.append({
                "validation_actions": action_str + label_suffix,
                "images": [image]
            })

        return current_image_logs

    # Run with original noise
    image_logs += run_sampling(init_z=None, label_suffix="", vae=vae, sampler=vis_sampler)

    # Run with init_noise if provided
    if init_noise is not None:
        init_noise = torch.clone(init_noise).to(device)
        image_logs += run_sampling(init_z=init_noise, label_suffix=" w/ init noise", vae=vae, sampler=vis_sampler)

    formatted_images = []
    for log in image_logs:
        images = log["images"]
        actions = log["validation_actions"]
        for image in images:
            formatted_images.append((actions, np.asarray(image)))

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for validation_actions, image in formatted_images:
                tracker.writer.add_images(validation_actions, image[None, ...], step, dataformats="NHWC")
        elif tracker.name == "wandb":
            import wandb

            wandb_images = []
            for validation_actions, image in formatted_images:
                wandb_images.append(wandb.Image(image, caption=validation_actions))
            tracker.log({"validation": wandb_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    def concatenate_images(image_caption, images_per_row=5, image_format="webp"):
        import io

        images = [log["images"][0] for log in image_caption]
        if images[0].size[0] > 1024:
            images = [image.resize((1024, 1024)) for image in images]

        widths, heights = zip(*(img.size for img in images))
        max_width = max(widths)
        total_height = sum(heights[i : i + images_per_row][0] for i in range(0, len(images), images_per_row))

        new_im = Image.new("RGB", (max_width * images_per_row, total_height))

        y_offset = 0
        for i in range(0, len(images), images_per_row):
            row_images = images[i : i + images_per_row]
            x_offset = 0
            for img in row_images:
                new_im.paste(img, (x_offset, y_offset))
                x_offset += max_width
            y_offset += heights[i]
        webp_image_bytes = io.BytesIO()
        new_im.save(webp_image_bytes, format=image_format)
        webp_image_bytes.seek(0)
        new_im = Image.open(webp_image_bytes)

        return new_im

    if config.train.local_save_vis:
        file_format = "webp"
        local_vis_save_path = osp.join(config.work_dir, "log_vis")
        os.umask(0o000)
        os.makedirs(local_vis_save_path, exist_ok=True)
        concatenated_image = concatenate_images(image_logs, images_per_row=5, image_format=file_format)
        save_path = (
            osp.join(local_vis_save_path, f"vis_{step}.{file_format}")
            if init_noise is None
            else osp.join(local_vis_save_path, f"vis_{step}_w_init.{file_format}")
        )
        concatenated_image.save(save_path)

    del vae
    torch.cuda.empty_cache()
    flush()
    return image_logs


def train(config, args, accelerator, model, optimizer, lr_scheduler, train_dataloader, train_diffusion, logger, vae=None):
    if getattr(config.train, "debug_nan", False):
        DebugUnderflowOverflow(model)
        logger.info("NaN debugger registered. Start to detect overflow during training.")
    log_buffer = LogBuffer()

    global_step = start_step + 1
    skip_step = max(config.train.skip_step, global_step) % train_dataloader_len
    skip_step = skip_step if skip_step < (train_dataloader_len - 20) else 0
    loss_nan_timer = 0
    logger.info("Start training...")
    # Cache Dataset for BatchSampler
    if args.caching and config.model.multi_scale:
        caching_start = time.time()
        logger.info(
            f"Start caching your dataset for batch_sampler at {cache_file}. \n"
            f"This may take a lot of time...No training will launch"
        )
        train_dataloader.batch_sampler.sampler.set_start(max(train_dataloader.batch_sampler.exist_ids, 0))
        accelerator.wait_for_everyone()
        for index, _ in enumerate(train_dataloader):
            accelerator.wait_for_everyone()
            if index % 2000 == 0:
                logger.info(
                    f"rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
                print(
                    f"rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
            if (time.time() - caching_start) / 3600 > 3.7:
                json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
                accelerator.wait_for_everyone()
                break
            if len(train_dataloader.batch_sampler.cached_idx) == len(train_dataloader) - 1000:
                logger.info(
                    f"Saving rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
                json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
            accelerator.wait_for_everyone()
            continue
        accelerator.wait_for_everyone()
        print(f"Saving rank-{rank} Cached file len: {len(train_dataloader.batch_sampler.cached_idx)}")
        json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
        return

    # Now you train the model
    for epoch in range(start_epoch, config.train.num_epochs):
        # Set epoch for regular datasets with samplers
        if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        if train_dataloader_len is not None:
            progress_bar = tqdm(
                total=train_dataloader_len,
                disable=not accelerator.is_local_main_process,
                position=0,
                leave=True,
            )
        else:
            # For IterableDataset, we'll update based on steps instead
            progress_bar = tqdm(
                disable=not accelerator.is_local_main_process,
                position=0,
                leave=True,
            )
        time_start, last_tic = time.time(), time.time()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            accelerator.wait_for_everyone()
            data_time_all = 0
            model_time_all = 0
            vae_time_all = 0
            data_time_start = time.time()

            # Get next batch
            clean_images = batch['img'].to(accelerator.device)  # Already VAE encoded by dataset
            obs = batch['obs'].to(accelerator.device)
            data_info = batch['data_info']

            # Get action conditioning
            y = batch['y'].to(accelerator.device)  # [B, seq_len, 5] one-hot encoded actions
            y_mask = batch['y_mask'].to(accelerator.device)  # [B, 1, seq_len] attention mask

            grad_norm = None
            accelerator.wait_for_everyone()
            model_time_start = time.time()
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term = train_diffusion.training_losses(
                    model, clean_images, torch.randint(0, config.scheduler.train_sampling_steps, (clean_images.shape[0],), device=clean_images.device).long(), 
                    model_kwargs=dict(y=y, mask=y_mask, data_info=data_info, obs=obs)
            )
                loss = loss_term["loss"].mean()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
            optimizer.step()
            lr_scheduler.step()
            accelerator.wait_for_everyone()
            model_time_all += time.time() - model_time_start

            if torch.any(torch.isnan(loss)):
                loss_nan_timer += 1
            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.train.log_interval == 0 or (step + 1) == 1:
                accelerator.wait_for_everyone()
                t = (time.time() - last_tic) / config.train.log_interval
                t_d = data_time_all / config.train.log_interval
                t_m = model_time_all / config.train.log_interval
                t_v = vae_time_all / config.train.log_interval
                avg_time = (time.time() - time_start) / (step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                
                # Handle ETA calculation for both IterableDataset and regular Dataset
                if isinstance(train_dataloader.dataset, torch.utils.data.IterableDataset):
                    current_step = step + 1
                    eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (train_dataloader_len - step - 1))))
                else:
                    current_step = (global_step - sampler.step_start // config.train.train_batch_size) % train_dataloader_len
                    current_step = train_dataloader_len if current_step == 0 else current_step
                    eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step - 1))))

                log_buffer.average()

                info = (
                    f"Epoch: {epoch} | Global Step: {global_step} | Local Step: {current_step} // {train_dataloader_len}, "
                    f"total_eta: {eta}, epoch_eta:{eta_epoch}, time: all:{t:.3f}, model:{t_m:.3f}, data:{t_d:.3f}, vae:{t_v:.3f}, "
                    f"lr:{lr:.3e}, "
                )
                info += (
                    f"s:({model.module.sana.h}, {model.module.sana.w}), "
                    if hasattr(model, "module")
                    else f"s:({model.h}, {model.w}), "
                )

                info += ", ".join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
                model_time_all = 0
                vae_time_all = 0
                if accelerator.is_main_process:
                    logger.info(info)

            logs.update(lr=lr)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            global_step += 1

            if loss_nan_timer > 20:
                raise ValueError("Loss is NaN too much times. Break here.")
            if global_step % config.train.save_model_steps == 0: # or (time.time() - training_start_time) / 3600 > 3.8:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    ckpt_saved_path = save_checkpoint(
                        osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        step=global_step,
                        model=accelerator.unwrap_model(model),
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        generator=generator,
                        add_symlink=True,
                        keep_last=True
                    )
                    if config.train.online_metric and global_step % config.train.eval_metric_step == 0 and step > 1:
                        online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                        os.makedirs(online_metric_monitor_dir, exist_ok=True)
                        with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                            f.write(osp.join(config.work_dir, "config.py") + "\n")
                            f.write(ckpt_saved_path)

                # if (time.time() - training_start_time) / 3600 > 3.8:
                #     logger.info(f"Stopping training at epoch {epoch}, step {global_step} due to time limit.")
                #     return
            if config.train.visualize and (global_step % config.train.eval_sampling_steps == 0 or (step + 1) == 1):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if validation_noise is not None:
                        log_validation(
                            accelerator=accelerator,
                            config=config,
                            model=model,
                            logger=logger,
                            step=global_step,
                            device=accelerator.device,
                            vae=vae,
                            init_noise=validation_noise,
                        )
                    else:
                        log_validation(
                            accelerator=accelerator,
                            config=config,
                            model=model,
                            logger=logger,
                            step=global_step,
                            device=accelerator.device,
                            vae=vae,
                        )

            # avoid dead-lock of multiscale data batch sampler
            # for internal, refactor dataloader logic to remove the ad-hoc implementation
            if (
                config.model.multi_scale
                and (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step) < 30
            ):
                global_step = epoch * train_dataloader_len
                logger.info("Early stop current iteration")
                break

            data_time_start = time.time()

        if epoch % config.train.save_model_epochs == 0 or epoch == config.train.num_epochs and not config.debug:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # os.umask(0o000)
                ckpt_saved_path = save_checkpoint(
                    osp.join(config.work_dir, "checkpoints"),
                    epoch=epoch,
                    step=global_step,
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    generator=generator,
                    add_symlink=True,
                )

                online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                os.makedirs(online_metric_monitor_dir, exist_ok=True)
                with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                    f.write(osp.join(config.work_dir, "config.py") + "\n")
                    f.write(ckpt_saved_path)
        accelerator.wait_for_everyone()


@pyrallis.wrap()
def main(cfg: SanaConfig) -> None:
    global train_dataloader_len, start_epoch, start_step, vae, generator, num_replicas, rank, training_start_time
    global load_vae_feat, load_text_feat, validation_noise, text_encoder, tokenizer
    global max_length, validation_prompts, latent_size, valid_prompt_embed_suffix, null_embed_path
    global image_size, cache_file, total_steps
    import itertools  # Add this import for random sampling

    config = cfg
    args = cfg
    # config = read_config(args.config)

    training_start_time = time.time()
    load_from = True
    if args.resume_from or config.model.resume_from:
        load_from = False
        config.model.resume_from = dict(
            checkpoint=args.resume_from or config.model.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True,
        )

    # if args.debug:
    #     config.train.log_interval = 1
    #     config.train.train_batch_size = min(64, config.train.train_batch_size)
    #     args.report_to = "tensorboard"

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.train.use_fsdp:
        init_train = "FSDP"
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig

        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        )
    else:
        init_train = "DDP"
        fsdp_plugin = None

    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=osp.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[init_handler],
    )

    log_name = "train_log.log"
    logger = get_root_logger(osp.join(config.work_dir, log_name))
    logger.info(accelerator.state)

    config.train.seed = init_random_seed(getattr(config.train, "seed", None))
    set_random_seed(config.train.seed + int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)

    if accelerator.is_main_process:
        pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)
        if args.report_to == "wandb":
            import wandb
            load_dotenv()
            wandb.login(key=os.getenv('WANDB_API_KEY'))
            wandb.init(
                project=args.tracker_project_name,
                resume="allow",
                name=None  # Let wandb generate a name
            )
    logger.info(f"Config: \n{config}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.train.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.model.image_size
    latent_size = int(image_size) // config.vae.vae_downsample_rate
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    vae = None
    validation_noise = (
        torch.randn(config.train.train_batch_size, 3, image_size, image_size, device="cpu", generator=generator)
        if getattr(config.train, "deterministic_validation", False)
        else None
    )
    if not config.data.load_vae_feat:
        vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, accelerator.device).to(torch.float16)
    logger.info(f"vae type: {config.vae.vae_type}")

    os.makedirs(config.train.null_embed_root, exist_ok=True)
    null_embed_path = osp.join(
        config.train.null_embed_root,
        f"null_embed_diffusers_{config.vae.vae_type}_{latent_size}.pth",
    )

    # 1. build scheduler
    train_diffusion = Scheduler(
        str(config.scheduler.train_sampling_steps),
        noise_schedule=config.scheduler.noise_schedule,
        predict_v=config.scheduler.predict_v,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.train.snr_loss,
        flow_shift=config.scheduler.flow_shift,
    )
    predict_info = f"v-prediction: {config.scheduler.predict_v}, noise schedule: {config.scheduler.noise_schedule}"
    if "flow" in config.scheduler.noise_schedule:
        predict_info += f", flow shift: {config.scheduler.flow_shift}"
    if config.scheduler.weighting_scheme in ["logit_normal", "mode"]:
        predict_info += (
            f", flow weighting: {config.scheduler.weighting_scheme}, "
            f"logit-mean: {config.scheduler.logit_mean}, logit-std: {config.scheduler.logit_std}"
        )
    logger.info(predict_info)

    # 2. build models
    model_kwargs = {
        "pe_interpolation": config.model.pe_interpolation,
        "config": config,
        "qk_norm": config.model.qk_norm,
        "micro_condition": config.model.micro_condition,
        "y_norm": True,
        "attn_type": config.model.attn_type,
        "ffn_type": config.model.ffn_type,
        "mlp_ratio": config.model.mlp_ratio,
        "mlp_acts": list(config.model.mlp_acts),
        "in_channels": config.model.in_channels,
        "y_norm_scale_factor": 0.01,
        "use_pe": config.model.use_pe,
        "linear_head_dim": config.model.linear_head_dim,
        "pred_sigma": pred_sigma,
        "learn_sigma": learn_sigma,
        "caption_channels": config.model.num_classes,
        "model_max_length": config.data.sequence_length-1,
        "seq_length": config.data.sequence_length,
        "vae": vae,
    }
    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        getattr(config.model, "fp32_attention", False),
        input_size=latent_size,
        **model_kwargs,
    ).train()
    logger.info(
        colored(
            f"{model.__class__.__name__}:{config.model.model}, "
            f"Total Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, "
            f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M",
            "green",
            attrs=["bold"],
        )
    )
    # 2-1. load model
    if args.load_from is not None:
        config.model.load_from = args.load_from
    if config.model.load_from is not None and load_from:
        _, missing, unexpected, _ = load_checkpoint(
            config.model.load_from,
            model,
            load_ema=config.model.resume_from.get("load_ema", False),
            null_embed_path=null_embed_path,
        )
        logger.warning(f"Missing keys: {missing}")
        logger.warning(f"Unexpected keys: {unexpected}")

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # 3. build dataloader
    config.data.data_dir = config.data.data_dir if isinstance(config.data.data_dir, list) else [config.data.data_dir]
    config.data.data_dir = [
        data if data.startswith(("https://", "http://", "gs://", "/", "~")) else osp.abspath(osp.expanduser(data))
        for data in config.data.data_dir
    ]
    train_dataset = build_dataset(
        asdict(config.data),
        resolution=image_size,
        aspect_ratio_type=config.model.aspect_ratio_type,
        vae_downsample_rate=config.vae.vae_downsample_rate,
        vae=None,  # Pass VAE model to dataset
    )
    accelerator.wait_for_everyone()

    # Get distributed training info
    num_replicas = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    # Handle multi-scale training with aspect ratio batching
    if isinstance(train_dataset, torch.utils.data.IterableDataset):
        train_dataloader = build_dataloader(
            train_dataset,
            num_workers=config.train.num_workers,
            batch_size=config.train.train_batch_size,
            shuffle=False,  # Shuffle is handled by the dataset itself
        )
        train_dataloader_len = len(train_dataset)  # Length will be infinite for streaming dataset
    else:
        if config.model.multi_scale:
            drop_last = True
            uuid = hashlib.sha256("-".join(config.data.data_dir).encode()).hexdigest()[:8]
            cache_dir = osp.expanduser(f"~/.cache/_wids_batchsampler_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            base_pattern = (
                f"{cache_dir}/{getpass.getuser()}-{uuid}-sort_dataset{config.data.sort_dataset}"
                f"-hq_only{config.data.hq_only}-valid_num{config.data.valid_num}"
                f"-aspect_ratio{len(train_dataset.aspect_ratio)}-droplast{drop_last}"
                f"dataset_len{len(train_dataset)}"
            )
            cache_file = f"{base_pattern}-num_replicas{num_replicas}-rank{rank}"
            for i in config.data.data_dir:
                cache_file += f"-{i}"
            cache_file += ".json"

            # Set up distributed sampler with aspect ratio batching
            sampler = DistributedRangedSampler(train_dataset, num_replicas=num_replicas, rank=rank)
            batch_sampler = AspectRatioBatchSampler(
                sampler=sampler,
                dataset=train_dataset,
                batch_size=config.train.train_batch_size,
                aspect_ratios=train_dataset.aspect_ratio,
                drop_last=drop_last,
                ratio_nums=train_dataset.ratio_nums,
                config=config,
                valid_num=config.data.valid_num,
                hq_only=config.data.hq_only,
                cache_file=cache_file,
                caching=args.caching,
            )
            train_dataloader = build_dataloader(train_dataset, batch_sampler=batch_sampler, num_workers=config.train.num_workers)
            train_dataloader_len = len(train_dataloader)
            logger.info(f"rank-{rank} Cached file len: {len(train_dataloader.batch_sampler.cached_idx)}")
        else:
            # Standard distributed sampling
            sampler = DistributedRangedSampler(train_dataset, num_replicas=num_replicas, rank=rank)
            train_dataloader = build_dataloader(
                train_dataset,
                num_workers=config.train.num_workers,
                batch_size=config.train.train_batch_size,
                shuffle=False,
                sampler=sampler,
            )
            train_dataloader_len = len(train_dataloader)

    # Check dataset properties
    load_vae_feat = getattr(train_dataloader.dataset, "load_vae_feat", False)

    # 4. build optimizer and lr scheduler
    lr_scale_ratio = 1
    if getattr(config.train, "auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train.train_batch_size * get_world_size() * config.train.gradient_accumulation_steps,
            config.train.optimizer,
            **config.train.auto_lr,
        )
    optimizer = build_optimizer(model, config.train.optimizer)
    if config.train.lr_schedule_args and config.train.lr_schedule_args.get("num_warmup_steps", None):
        config.train.lr_schedule_args["num_warmup_steps"] = (
            config.train.lr_schedule_args["num_warmup_steps"] * num_replicas
        )
    lr_scheduler = build_lr_scheduler(config.train, optimizer, train_dataloader, lr_scale_ratio)
    logger.warning(
        f"{colored(f'Basic Setting: ', 'green', attrs=['bold'])}"
        f"lr: {config.train.optimizer['lr']:.5f}, bs: {config.train.train_batch_size}, gc: {config.train.grad_checkpointing}, "
        f"gc_accum_step: {config.train.gradient_accumulation_steps}, qk norm: {config.model.qk_norm}, "
        f"fp32 attn: {config.model.fp32_attention}, attn type: {config.model.attn_type}, ffn type: {config.model.ffn_type}, "
        f"precision: {config.model.mixed_precision}"
    )

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    total_steps = len(train_dataloader) * config.train.num_epochs

    # Resume training
    if config.model.resume_from is not None and config.model.resume_from["checkpoint"] is not None:
        rng_state = None
        ckpt_path = osp.join(config.work_dir, "checkpoints")
        check_flag = osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0
        if config.model.resume_from["checkpoint"] == "latest":
            if check_flag:
                checkpoints = os.listdir(ckpt_path)
                if "latest.pth" in checkpoints and osp.exists(osp.join(ckpt_path, "latest.pth")):
                    config.model.resume_from["checkpoint"] = osp.realpath(osp.join(ckpt_path, "latest.pth"))
                else:
                    checkpoints = [i for i in checkpoints if i.startswith("epoch_")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.replace(".pth", "").split("_")[3]))
                    config.model.resume_from["checkpoint"] = osp.join(ckpt_path, checkpoints[-1])
            else:
                config.model.resume_from["checkpoint"] = config.model.load_from

        if config.model.resume_from["checkpoint"] is not None:
            _, missing, unexpected, rng_state = load_checkpoint(
                **config.model.resume_from,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                null_embed_path=null_embed_path,
            )

            logger.warning(f"Missing keys: {missing}")
            logger.warning(f"Unexpected keys: {unexpected}")

            path = osp.basename(config.model.resume_from["checkpoint"])
        try:
            start_epoch = int(path.replace(".pth", "").split("_")[1]) - 1
            start_step = int(path.replace(".pth", "").split("_")[3])
        except:
            pass

        # resume randomise
        if rng_state:
            logger.info("resuming randomise")
            torch.set_rng_state(rng_state["torch"])
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
            np.random.set_state(rng_state["numpy"])
            random.setstate(rng_state["python"])
            generator.set_state(rng_state["generator"])  # resume generator status

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    # Start Training
    train(
        config=config,
        args=args,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        train_diffusion=train_diffusion,
        logger=logger,
        vae=vae,
    )


# Global variables for validation
val_dataset = None
val_dataloader = None
val_iterator = None

def random_sample_from_iterable(dataset, batch_size=1, upper_limit=100):
    samples = []
    for _ in range(batch_size):
        skip = random.randint(0, upper_limit)
        sample = next(itertools.islice(dataset, skip, None))
        samples.append(sample)
    
    obs = torch.stack([s['obs'] for s in samples])
    img = torch.stack([s['img'] for s in samples])
    y = torch.cat([s['y'] for s in samples])
    y_mask = torch.cat([s['y_mask'] for s in samples])
    data_info = [s['data_info'] for s in samples]
    
    # Reshape img and obs to [b, c, h, w]
    img = img.squeeze(0)
    obs = obs.squeeze(0)
    
    return {
        'obs': obs,
        'img': img,
        'y': y,
        'y_mask': y_mask,
        'data_info': data_info
    }

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
