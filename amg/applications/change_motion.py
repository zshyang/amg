"""
render the vidoe by changing the motion of the human

author
    zhangsihao yang
"""
import argparse
import os
from glob import glob
from typing import Any

import cv2
import imageio
import torch
import yaml
from amg.tools.datasets.motion_video_feature_dataset import \
    MotionVideoFeatureDataset
from amg.tools.modules.autoencoder import AutoencoderKL
from amg.tools.modules.clip_embedder import FrozenOpenCLIPEmbedderZero
from amg.tools.modules.diffusions.diffusion_ddim import DiffusionDDIM
from amg.tools.modules.unet.unet_lora import UNetSD_LoRA
from easydict import EasyDict
from einops import rearrange
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.util import to_device


def load_cfg(cfg_path) -> Any:
    """
    load config based on args
    """
    with open(cfg_path, 'r') as f:
        cfg: dict = yaml.load(f.read(), Loader=yaml.SafeLoader)

    # convert cfg to EasyDict
    cfg = EasyDict(cfg)

    return cfg


def encode_conditions(clip_encoder, captions):
    with torch.no_grad():
        y = clip_encoder(captions).detach()
        zero_y = clip_encoder("").detach()

    return y, zero_y


def load_autoencoder(cfg):
    """
    Load and initialize the AutoencoderKL model from the given configuration.

    Args:
        cfg (EasyDict or dict): Configuration dictionary containing the parameters for the AutoencoderKL.

    Returns:
        AutoencoderKL: The initialized autoencoder model with its parameters set to not require gradients.
    """
    autoencoder = AutoencoderKL(**cfg.auto_encoder)
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    return autoencoder


def load_unet(cfg):
    if cfg.concat_input:
        cfg.UNet.in_dim = 8
    model = UNetSD_LoRA(**cfg.UNet).eval()

    load_dict = torch.load(cfg.infer_checkpoint, map_location='cpu')

    if 'state_dict' in load_dict.keys():
        load_dict = load_dict['state_dict']
    _ = model.load_state_dict(load_dict, strict=True)

    torch.cuda.empty_cache()

    return model.cuda()


def safe_pop(d, key, default=None):
    if key in d:
        value = d[key]
        del d[key]
    else:
        value = default
    return value


def setup_data(cfg):
    """
    setup the data
    """
    dataset_type = safe_pop(cfg.vid_dataset, "type", "")

    # create dataset
    if dataset_type == "MotionVideoFeatureDataset":
        dataset = MotionVideoFeatureDataset(
            **cfg.vid_dataset,
            sample_fps=cfg.sample_fps,
            max_frames=cfg.max_frames,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # create dataloader
    dataloader_kwargs = {
        'batch_size': cfg.batch_sizes[str(cfg.max_frames)],
    }
    cuda_kwargs = {
        'num_workers': 10,
        'pin_memory': True,
        'shuffle': False
    }
    dataloader_kwargs.update(cuda_kwargs)
    loader = DataLoader(dataset, **dataloader_kwargs)

    return loader


@torch.no_grad()
def visualize(cfg, batch, diffusion, model, clip_encoder):
    """
    visualize the model
    """
    batch = to_device(batch, 0, non_blocking=True)
    (
        gh_video_feature_data,
        video_feature_data,
        _,
        zero_y,
    ) = batch
    gh_video_feature_data = rearrange(
        gh_video_feature_data, 'b f c h w -> b c f h w',
    )

    caption_y, _ = encode_conditions(clip_encoder, cfg.new_caption)

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

    # decode the video data
    return video_data, gh_video_feature_data, video_feature_data


def decode_to_image(latent_code, cfg, autoencoder):
    batch_size, _, _, _, _ = latent_code.shape

    decoder_bs = 8

    latent_code = 1. / cfg.scale_factor * latent_code  # [64, 4, 32, 48]
    latent_code = rearrange(latent_code, 'b c f h w -> (b f) c h w')

    chunk_size = min(decoder_bs, latent_code.shape[0])
    video_data_list = torch.chunk(
        latent_code,
        latent_code.shape[0]//chunk_size,
        dim=0
    )
    decode_data = []
    for vd_data in video_data_list:
        gen_frames = autoencoder.decode(vd_data)
        decode_data.append(gen_frames)
    video_data = torch.cat(decode_data, dim=0)
    video_data = rearrange(
        video_data,
        '(b f) c h w -> b c f h w',
        b=batch_size,
    )

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    vid_mean = torch.tensor(
        mean, device=video_data.device
    ).view(
        1, -1, 1, 1, 1
    )  # ncfhw
    vid_std = torch.tensor(
        std, device=video_data.device
    ).view(
        1, -1, 1, 1, 1
    )  # ncfhw

    gen_video = video_data.mul_(vid_std).add_(vid_mean)  # 8x3x16x256x384
    gen_video.clamp_(0, 1)
    gen_video = gen_video * 255.0

    return gen_video


def decode_to_list_video(video_data, video_feature_data, cfg, autoencoder, save_path):
    """
    """
    gen_video = decode_to_image(video_data, cfg, autoencoder)
    ori_video = decode_to_image(video_feature_data, cfg, autoencoder)

    gen_video = rearrange(gen_video, 'b c f h w -> b f h w c')
    ori_video = rearrange(ori_video, 'b c f h w -> b f h w c')

    nrow = max(int(video_data.size(0) / 2), 1)

    images = torch.cat([gen_video, ori_video], dim=3)
    images = rearrange(images, '(r j) f h w c -> f (r h) (j w) c', r=nrow)
    images = [(img.cpu().numpy()).astype('uint8') for img in images]

    # setup the path
    frame_dir = os.path.join(
        os.path.dirname(save_path),
        '%s_frames' % (os.path.basename(save_path))
    )
    os.system(f'rm -rf {frame_dir}')
    os.makedirs(frame_dir, exist_ok=True)

    # write the images
    for fid, frame in enumerate(images):
        tpth = os.path.join(frame_dir, '%04d.png' % (fid+1))
        cv2.imwrite(
            tpth, frame[:, :, ::-1],
            [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        )

    # Create video writer
    save_fps = 4
    writer = imageio.get_writer(
        save_path, fps=save_fps, codec='libx264', quality=10
    )

    # Read and write each frame
    list_png_path = glob(os.path.join(frame_dir, '*.png'))
    list_png_path.sort()
    for frame_file in list_png_path:
        frame = imageio.imread(frame_file)
        writer.append_data(frame)

    # Close the writer
    writer.close()


def main():
    """
    main function used to change the  motion of the human
    """
    parser = argparse.ArgumentParser(description="PyTorch gh2v")

    parser.add_argument('--cfg', type=str, required=True,
                        help='input path for the config file')

    args = parser.parse_args()
    cfg: Any = load_cfg(args.cfg)

    # load the dataset
    dataloader = setup_data(cfg)

    print("Loading CLIP...")
    clip_encoder = FrozenOpenCLIPEmbedderZero(**cfg.embedder)
    clip_encoder.model.cuda()

    print("Loading Autoencoder...")
    autoencoder = load_autoencoder(cfg).cuda()

    # load a condition video
    model = load_unet(cfg)

    # create a new text
    for index, batch in tqdm(enumerate(dataloader)):
        # generate a new video
        diffusion = DiffusionDDIM(**cfg.Diffusion)
        (
            video_data, _, video_feature_data,
        ) = visualize(cfg, batch, diffusion, model, clip_encoder)

        # save the new video
        decode_to_list_video(
            video_data, video_feature_data, cfg, autoencoder,
            os.path.join(cfg.save_dir, f"{index:04d}.mp4"),
        )

    print("Change Motion Done!")


if __name__ == "__main__":
    main()
