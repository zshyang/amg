"""
author
    zhangsihao yang
logs
    2024-07-27
        file created
"""
import argparse
import functools
import os
import os.path as osp
import time
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from easydict import EasyDict
from einops import rearrange
from tools.basic_funcs.pretrain_functions import pretrain_instructvideo
from tools.datasets.gh_video_feature_dataset import GHVideoFeatureDataset
from tools.datasets.video_feature_dataset import VideoFeatureDataset
from tools.modules.diffusions.diffusion_ddim import DiffusionDDIM
from tools.modules.unet.unet_lora import (BasicTransformerBlock, LoRA,
                                          UNetSD_LoRA)
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.streams import Event
# from torch.distributed.fsdp.api import (BackwardPrefetch, CPUOffload,
#                                         ShardingStrategy)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch, CPUOffload)
from torch.distributed.fsdp.fully_sharded_data_parallel import \
    FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import (_or_policy, enable_wrap,
                                         size_based_auto_wrap_policy,
                                         transformer_auto_wrap_policy, wrap)
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from transformers.models.t5.modeling_t5 import T5Block
from utils.optim import AnnealingLR
from utils.util import to_device


def lambda_auto_wrap_policy(
    module: nn.Module, recurse: bool, nonwrapped_numel: int, lambda_fn: Callable
) -> bool:
    """
    A convenient auto wrap policy to wrap submodules based on an arbitrary user
    function. If `lambda_fn(submodule) == True``, the submodule will be wrapped as
    a `wrapper_cls` unit.

    Return if a module should be wrapped during auto wrapping.

    The first three parameters are required by :func:`_recursive_wrap`.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.

        lambda_fn (Callable[[nn.Module], bool]): If this returns ``True``, then
            this module will be wrapped.
    """
    if recurse:
        return True  # always recurse
    return lambda_fn(module)


def rank_print(print_str, rank, file=None):
    if rank == 0:
        print(print_str)
        if file is not None:
            print(print_str, file=open(file, "a"), flush=True)


def load_cfg(cfg_path) -> Any:
    """
    load config based on args
    """
    with open("configs/base.yaml", 'r') as f:
        base_cfg: dict = yaml.load(f.read(), Loader=yaml.SafeLoader)

    with open(cfg_path, 'r') as f:
        cfg: dict = yaml.load(f.read(), Loader=yaml.SafeLoader)

    # merge these two configs, and cfg will overwrite base_cfg
    cfg = {**base_cfg, **cfg}

    # convert cfg to EasyDict
    cfg = EasyDict(cfg)

    cfg.cfg_file = os.path.split(cfg_path)[1]

    return cfg


def setup_environment(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def safe_pop(d, key, default=None):
    if key in d:
        value = d[key]
        del d[key]
    else:
        value = default
    return value


def setup_data(cfg, rank, world_size):
    """
    setup the data
    """
    dataset_type = safe_pop(cfg.vid_dataset, "type", "")

    # create dataset
    if dataset_type == "GHVideoFeatureDataset":
        dataset = GHVideoFeatureDataset(
            **cfg.vid_dataset,
            sample_fps=cfg.sample_fps,
            max_frames=cfg.max_frames,
        )
    else:
        dataset = VideoFeatureDataset(
            **cfg.vid_dataset,
            sample_fps=cfg.sample_fps,
            max_frames=cfg.max_frames,
        )

    # create sampler
    sampler = DistributedSampler(
        dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True,
    )

    # create dataloader
    dataloader_kwargs = {
        'batch_size': cfg.batch_sizes[str(cfg.max_frames)],
        'sampler': sampler,
    }
    cuda_kwargs = {
        'num_workers': 2,
        'pin_memory': True,
        'shuffle': False
    }
    dataloader_kwargs.update(cuda_kwargs)
    loader = DataLoader(dataset, **dataloader_kwargs)

    return loader


def freeze_all_except_lora(model):
    if hasattr(model, 'module'):
        tmp_model = 'model.module'
    else:
        tmp_model = 'model'

    for name, param in eval(tmp_model).named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


# Wrap the model using LoRA policy from llama-recipes or custom policy:
# This checks for lora layers (has weight and requires_grad)


def modify_with_instruct_pix2pix(model):
    # Assuming unet and in_channels are already defined
    in_channels = 8
    out_channels = model.input_blocks[0][0].out_channels

    # Define a new convolutional layer with the same parameters as the original
    new_conv_in = nn.Conv2d(
        in_channels,
        out_channels,
        model.input_blocks[0][0].kernel_size,
        model.input_blocks[0][0].stride,
        model.input_blocks[0][0].padding,
        bias=True,

    )

    # Copy the weights from the original convolutional layer to the new one
    with torch.no_grad():
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(model.input_blocks[0][0].weight)

        assert new_conv_in.bias is not None
        new_conv_in.bias.copy_(model.input_blocks[0][0].bias)

    # Set the requires_grad attribute appropriately
    new_conv_in.weight.requires_grad = True
    new_conv_in.bias.requires_grad = True

    # Replace the original conv_in with the new one
    model.input_blocks[0][0] = new_conv_in.to(
        model.input_blocks[0][0].weight.device
    )

    return model


def setup_model(cfg, rank, optimizer=None, scaler=None):
    # my implementation
    model = UNetSD_LoRA(**cfg.UNet)

    if cfg.UNet.use_lora:
        model = freeze_all_except_lora(model)

    # load the pretrain model
    resume_step = 1
    if cfg.Pretrain.resume_checkpoint != "":
        if cfg.Pretrain.loaded_resume_checkpoint != "" and cfg.concat_input:
            model = modify_with_instruct_pix2pix(model)
        model, resume_step = pretrain_instructvideo(
            **cfg.Pretrain, model=model,
            optimizer=optimizer, scaler=scaler,
            grad_scale=cfg.grad_scale
        )
    else:
        rank_print("Train from scratch.", rank)

    torch.cuda.set_device(rank)

    if cfg.concat_input and cfg.Pretrain.loaded_resume_checkpoint == "":
        model = modify_with_instruct_pix2pix(model)

    # if cfg.Pretrain.loaded_resume_checkpoint != "":
    #     optimizer.load_state_dict(
    #         torch.load(cfg.Pretrain.loaded_resume_checkpoint)["optimizer"]
    #     )
    #     model.load_state_dict(
    #         torch.load(cfg.Pretrain.loaded_resume_checkpoint)['state_dict']
    #     )

    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    return model, resume_step


def setup_optimizer_and_scheduler(cfg, model):
    """
    setup the optimizer and scheduler
    """
    # optimizer = optim.Adadelta(model.parameters(), lr=cfg.lr)

    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    scaler = GradScaler(enabled=cfg.use_fp16)

    scheduler = AnnealingLR(
        optimizer=optimizer,
        base_lr=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.num_steps,
        decay_mode=cfg.decay_mode
    )

    return optimizer, scheduler, scaler


def train(
    cfg, model, diffusion, optimizer, batch, rank, world_size, step, scaler,
):
    """
    training the model
    """
    model.train()
    ddp_loss = torch.zeros(2).to(rank)

    batch = to_device(batch, rank, non_blocking=True)
    (
        gh_video_feature_data, video_feature_data, caption_y,
        video_key, zero_y,
    ) = batch

    gh_video_feature_data = rearrange(
        gh_video_feature_data, 'b f c h w -> b c f h w',
    )
    if not cfg.concat_input:
        gh_video_feature_data = None

    video_feature_data = rearrange(
        video_feature_data, 'b f c h w -> b c f h w',
    )

    # data, target = data.to(rank), target.to(rank)
    optimizer.zero_grad()

    model_kwargs_ddpm = {'y': caption_y}

    noise = torch.randn_like(
        video_feature_data,
        device=rank,
    )
    noise = noise.contiguous()

    # random generate time steps
    timesteps = torch.randint(
        0,
        diffusion.num_timesteps,
        (video_feature_data.size(0),),
        device=rank,
    )
    timesteps = timesteps.long()

    with autocast(enabled=cfg.use_fp16):
        loss_recon = diffusion.loss(
            x0=video_feature_data,
            t=timesteps,
            model=model,
            model_kwargs=model_kwargs_ddpm,
            noise=noise,
            use_div_loss=cfg.use_div_loss,
            gh_video_feature_data=gh_video_feature_data,
        )
        loss_recon = loss_recon.mean()

        ddp_loss[0] += loss_recon.item()
        ddp_loss[1] += len(caption_y)

        optimizer.zero_grad()
        scaler.scale(loss_recon).backward()
        scaler.step(optimizer)
        scaler.update()

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        avg_loss = ddp_loss[0] / ddp_loss[1]
        rank_print(
            f'Train Step: {step} \tLoss: {avg_loss:.6f}',
            rank,
            cfg.log_file,
        )


def obtain_time():
    cur_time = time.localtime()
    month = str(cur_time.tm_mon).zfill(2)
    day = str(cur_time.tm_mday).zfill(2)
    hour = str(cur_time.tm_hour).zfill(2)
    minute = str(cur_time.tm_min).zfill(2)
    str_time = f'{month}{day}-{hour}-{minute}'
    return str_time


@torch.no_grad()
def visualize(cfg, batch, rank, diffusion, model, step, world_size, resume_step):
    """
    visualize the model
    """

    if not (step == resume_step or step == cfg.num_steps or step % cfg.viz_interval == 0):
        return

    batch = to_device(batch, rank, non_blocking=True)
    (
        gh_video_feature_data, video_feature_data,
        caption_y, video_key, zero_y
    ) = batch
    gh_video_feature_data = rearrange(
        gh_video_feature_data, 'b f c h w -> b c f h w',
    )
    if not cfg.concat_input:
        gh_video_feature_data = None
    video_feature_data = rearrange(
        video_feature_data, 'b f c h w -> b c f h w',
    )

    with autocast(enabled=cfg.use_fp16):
        noise = torch.randn_like(video_feature_data)  # viz_num: 8

        model_kwargs = [
            {'y': caption_y},
            {'y': zero_y},
        ]
        video_data = diffusion.cond_ddim_sample_loop(
            noise=noise,
            model=model.eval(),  # .requires_grad_(False),
            model_kwargs=model_kwargs,
            guide_scale=cfg.guide_scale,
            ddim_timesteps=cfg.ddim_timesteps,
            eta=0.0,
            gh_video_data=gh_video_feature_data,
        )

    # save the generated videos.
    file_name = f'rank_{world_size:02d}-{rank:02d}.pth'
    local_path = os.path.join(
        cfg.temp_dir, f'sample_{step:06d}/{file_name}'
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    torch.save(
        [video_data, gh_video_feature_data, video_feature_data],
        local_path
    )


def save_model(step, cfg, resume_step, rank, model, optimizer, scaler):
    """
    save the model
    """
    if (step == cfg.num_steps or step % cfg.save_ckp_interval == 0) and resume_step != step:
        if rank == 0:
            local_key = osp.join(
                cfg.temp_dir,
                f'checkpoints/non_ema_{step:07d}.pth'
            )
            state_dict = {
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'step': step
            }

            os.makedirs(osp.dirname(local_key), exist_ok=True)
            torch.save(state_dict, local_key)


def fsdp_main(rank, world_size, cfg):
    """
    main function for running fsdp on gh2v
    """
    setup_environment(rank, world_size)

    # finish the dataset and dataloader
    dataloader = setup_data(cfg, rank, world_size,)

    # setup the model
    model, resume_step = setup_model(cfg, rank)
    rank_print(f"Model Loaded", rank)

    # setup the diffusion
    diffusion = DiffusionDDIM(**cfg.Diffusion)

    # setup the event
    init_start_event = Event(enable_timing=True)
    init_end_event = Event(enable_timing=True)

    # set optimizer and learning rate scheduler
    optimizer, scheduler, scaler = setup_optimizer_and_scheduler(cfg, model)
    model, resume_step = pretrain_instructvideo(
        **cfg.Pretrain, model=model,
        optimizer=optimizer, scaler=scaler,
        grad_scale=cfg.grad_scale
    )

    # start the training loop
    init_start_event.record()  # type: ignore
    rank_iter = iter(dataloader)
    for step in range(resume_step, cfg.num_steps + 1):
        rank_print(f"Step: {step}/{cfg.num_steps}", rank)

        batch = next(rank_iter, None)
        if batch is None:
            rank_iter = iter(dataloader)
            batch = next(rank_iter)

        train(
            cfg, model, diffusion, optimizer,
            batch, rank, world_size, step, scaler
        )
        visualize(
            cfg, batch, rank, diffusion, model, step, world_size, resume_step
        )
        save_model(step, cfg, resume_step, rank, model, optimizer, scaler)
        scheduler.step()

    # finish the training loop
    init_end_event.record()  # type: ignore
    rank_print(
        f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec",
        rank,
    )

    # clean up
    cleanup()


def setup_logging(cfg):
    """
    setup logging. create folder, create logging file.
    """
    # create the logging folder
    exp_name = '.'.join(
        osp.dirname(cfg.cfg_file).split("/")[1:] +
        [osp.basename(cfg.cfg_file).split('.')[0]]
    ) + '_' + obtain_time()
    cfg.temp_dir = osp.join(cfg.temp_dir, exp_name)
    os.makedirs(cfg.temp_dir, exist_ok=True)

    # create the logging file
    log_file = osp.join(cfg.temp_dir, 'log.txt')
    cfg.log_file = log_file


def main():
    """
    main function for running fsdp on gh2v
    """
    parser = argparse.ArgumentParser(description="PyTorch gh2v")

    parser.add_argument('--cfg', type=str, required=True,
                        help='input path for the config file')

    args = parser.parse_args()
    cfg: Any = load_cfg(args.cfg)

    setup_logging(cfg)

    torch.manual_seed(cfg.seed)

    WORLD_SIZE = torch.cuda.device_count()

    spawn(
        fsdp_main,
        args=(WORLD_SIZE, cfg),
        nprocs=WORLD_SIZE,
        join=True
    )


if __name__ == "__main__":
    main()
