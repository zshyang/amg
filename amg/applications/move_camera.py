"""
render the vidoe by changing the motion of the human
"""
import argparse
import os
from glob import glob
from typing import Any

import cv2
import imageio
import torch
import utils.transforms as data
import yaml
from easydict import EasyDict
from einops import rearrange
from PIL import Image
from tools.basic_funcs.pretrain_functions import pretrain_instructvideo
from tools.datasets.gh_video_feature_dataset import (GHVideoFeatureDataset,
                                                     get_valid_image_paths)
from tools.datasets.motion_video_feature_dataset import \
    MotionVideoFeatureDataset
from tools.modules.autoencoder import AutoencoderKL
from tools.modules.clip_embedder import FrozenOpenCLIPEmbedderZero
from tools.modules.diffusions.diffusion_ddim import DiffusionDDIM
from tools.modules.unet.unet_lora import UNetSD_LoRA
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, Dataset
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
    load_info = model.load_state_dict(load_dict, strict=True)

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
    if dataset_type == "GHVideoFeatureDataset":
        dataset = GHVideoFeatureDataset(
            **cfg.vid_dataset,
            sample_fps=cfg.sample_fps,
            max_frames=cfg.max_frames,
        )
    elif dataset_type == "MotionVideoFeatureDataset":
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
def visualize(
    cfg, decode_data, caption_y, zero_y, diffusion, model, clip_encoder
):
    """
    visualize the model
    """
    # batch = to_device(batch, 0, non_blocking=True)
    # (
    #     gh_video_feature_data, video_feature_data,
    #     _, video_key, zero_y
    # ) = batch
    gh_video_feature_data = rearrange(
        decode_data, 'b f c h w -> b c f h w',
    )

    # caption_y, _ = encode_conditions(clip_encoder, cfg.new_caption)

    if not cfg.concat_input:
        gh_video_feature_data = None
    video_feature_data = rearrange(
        decode_data, 'b f c h w -> b c f h w',
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


def decode_to_video(video_data, cfg, autoencoder, save_path):
    """
    """
    gen_video = decode_to_image(video_data, cfg, autoencoder)
    gen_video = rearrange(gen_video, 'b c f h w -> b f h w c')

    nrow = max(int(video_data.size(0) / 2), 1)
    images = rearrange(gen_video, '(r j) f h w c -> f (r h) (j w) c', r=nrow)
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
    # os.system(f'rm -rf {frame_dir}')


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
    # os.system(f'rm -rf {frame_dir}')


def load_image(image_path, transforms):
    # Load the image
    frame = cv2.imread(image_path)  # type: ignore

    # Convert the image from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore

    # Convert the frame to a PIL Image
    frame = Image.fromarray(frame)

    # Get the dimensions of the frame
    width, height = frame.size

    # Calculate the coordinates for the center crop
    new_height = 500
    left = 0
    upper = (height - new_height) // 2
    right = width
    lower = upper + new_height

    # Crop the frame to the center 500 pixels of height
    frame = frame.crop((left, upper, right, lower))

    # Convert the frame to a tensor
    frame = transforms([frame])

    return frame


def load_data(cfg, train_trans):
    """
    load the data and process with train_trans
    """
    list_path = glob(
        os.path.join(cfg.vid_dataset.gh_images_dir, "*.jpg")
    )
    list_path.sort()

    # hard code the original video fps
    original_fps = 24
    # compute the frame step
    frame_step = original_fps // cfg.sample_fps
    # get the valid image paths
    valid_images_feature_paths = get_valid_image_paths(
        list_path,
        frame_step,
        cfg.max_frames,
    )

    print(valid_images_feature_paths[cfg.vid_dataset.start_frame])

    list_frame_tensor = []
    for i, frame_index in enumerate(range(
        cfg.vid_dataset.start_frame,
        cfg.vid_dataset.start_frame + frame_step * cfg.max_frames,
        frame_step
    )):
        if len(valid_images_feature_paths) == 1:
            # this is for the case when only one clip is used
            frame = list_path[i]
        else:
            frame, _ = valid_images_feature_paths[frame_index]

        # load the frame
        frame_tensor = load_image(frame, train_trans)

        list_frame_tensor.append(frame_tensor)

    # convert the frame into tensor
    # video_data : [b, f, 3, h, w]
    video_data = torch.stack(list_frame_tensor, dim=1)

    # hard code the caption
    caption = "The main characters in the scene are a man and a woman, both sitting on a bench. The man is wearing a tie, and the woman is wearing a yellow dress. They appear to be engaged in a conversation, possibly discussing their feelings or sharing a moment of connection. The setting is a park at dusk, with the sun setting in the background. The atmosphere is calm and serene, with the warm hues of the sunset creating a peaceful ambiance. The scene is lit by the fading sunlight, casting a soft glow on the couple and the surrounding environment. The bench they are sitting on is positioned near a tree, which adds a natural element to the scene. In the background, there are a few other people scattered around the park, but they are not the main focus of the image. The couple seems to be enjoying their time together, sharing a quiet moment amidst the park's tranquility."

    # load zero_y
    zero_y = torch.load(cfg.vid_dataset.zero_y_path, map_location='cpu')

    return video_data, caption, zero_y


def encode_conditions(clip_encoder, captions):
    with torch.no_grad():
        y = clip_encoder(captions).detach()

    return y


def process_data(video_data, caption, zero_y, clip_encoder, autoencoder):
    scale_factor = 0.18215

    # decode_data : [16, 4, 32, 32]
    decode_data = autoencoder.encode_firsr_stage(
        video_data[0].cuda(), scale_factor
    ).detach()

    # y : [1, 77, 1024]
    y = encode_conditions(clip_encoder, [caption])

    return decode_data.unsqueeze(0), y, zero_y.cuda()


def main():
    """
    main function used to change the  motion of the human
    """
    parser = argparse.ArgumentParser(description="PyTorch hg2v")

    parser.add_argument('--cfg', type=str, required=True,
                        help='input path for the config file')

    args = parser.parse_args()
    cfg: Any = load_cfg(args.cfg)

    # load CLIP
    print("Loading CLIP...")
    clip_encoder = FrozenOpenCLIPEmbedderZero(**cfg.embedder)
    clip_encoder.model.cuda()

    # load the autoencoder
    print("Loading Autoencoder...")
    autoencoder = load_autoencoder(cfg).cuda()

    # load a condition video
    print("Loading UNet...")
    model = load_unet(cfg)

    # load the dataset
    print("Processing Data...")
    train_trans = data.Compose(
        [
            data.CenterCropWide(size=cfg.resolution),
            data.ToTensor(),
            data.Normalize(mean=cfg.mean, std=cfg.std)
        ]
    )
    video_data, caption, zero_y = load_data(cfg, train_trans)

    decode_data, y, zero_y = process_data(
        video_data, caption, zero_y, clip_encoder, autoencoder,
    )

    # generate a new video
    diffusion = DiffusionDDIM(**cfg.Diffusion)
    (
        video_data, gh_video_feature_data, video_feature_data,
    ) = visualize(
        cfg, decode_data, y, zero_y, diffusion, model, clip_encoder
    )

    # save the new video
    decode_to_list_video(
        video_data, video_feature_data, cfg, autoencoder,
        os.path.join(cfg.save_dir, f"{cfg.vid_dataset.start_frame:04d}.mp4"),
    )


if __name__ == "__main__":
    main()
