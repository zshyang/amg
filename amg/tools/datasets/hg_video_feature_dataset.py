"""
the video dataset with pre-extracted features

author
    zhangishao yang
logs
    2024-07-27
        file created
"""
import os
from copy import copy
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset


def get_next_image_path(image_path: str, step: int, num_step: int = 1) -> str:
    """
    Given an image path and a step size, compute the path of the next image.

    Args:
        image_path (str): The current image path.
            e.g. "../_data/processed/gh_images/frame_000000.jpg"
        step (int): The step size to move to the next image.

    Returns:
        str: The path of the next image.
    """
    # Extract the directory and file name from the path
    directory, file_name = os.path.split(image_path)

    # Extract the frame number from the file name
    base_name, ext = os.path.splitext(file_name)
    prefix, frame_number_str = base_name.split('_')
    frame_number = int(frame_number_str)

    # Compute the next frame number
    next_frame_number = frame_number + step * num_step

    # Construct the new file name with the incremented frame number
    # Format with leading zeros
    next_frame_number_str = f"{next_frame_number:06d}"
    next_file_name = f"{prefix}_{next_frame_number_str}{ext}"

    # Construct the full path for the next image
    next_image_path = os.path.join(directory, next_file_name)

    return next_image_path


def check_next_image_path(
    current_idx,
    step,
    next_image_path,
    idx_2_path,
    list_image_paths,
):
    """
    current_idx is always a valid index to search in the
    """
    for i in range(1, step + 1):
        if current_idx + i >= len(list_image_paths):
            return False

        possible_next_image_path = idx_2_path[current_idx + i]

        # find the next_image_path in the list_image_paths
        if possible_next_image_path == next_image_path:
            return True

    # next_image_path is not in the list_image_paths
    return False


def varify_image_path(
    image_path, step, path_2_idx, idx_2_path, max_frames: int,
    list_image_paths,
):
    """
    """
    next_image_path = image_path

    for _ in range(max_frames - 1):
        current_idx = path_2_idx[next_image_path]

        next_image_path = get_next_image_path(next_image_path, step)

        # check if the next image path is in the list of image paths
        if check_next_image_path(
            current_idx,
            step,
            next_image_path,
            idx_2_path,
            list_image_paths,
        ):
            continue
        else:
            return False
    return True


def get_middle_frame_path(image_path, step, max_frames):
    """
    """
    middle_frame_idx = max_frames // 2
    middle_frame_path = get_next_image_path(
        image_path, step, middle_frame_idx
    )

    return middle_frame_path


def get_valid_image_paths(list_image_paths, step, max_frames):
    """
    given a list of frames and the
    """
    # sort the list of image paths
    list_image_paths.sort()

    # create the mapping from image path to index
    path_2_idx = {}
    for i, image_path in enumerate(list_image_paths):
        path_2_idx[image_path] = i
    idx_2_path = {v: k for k, v in path_2_idx.items()}

    # initialize the list of valid image paths
    valid_image_paths = []

    for image_path in list_image_paths:
        # compute the list of step image
        exist_in_list_image_paths = varify_image_path(
            image_path,
            step,
            path_2_idx,
            idx_2_path,
            max_frames,
            list_image_paths,
        )
        # if yes,
        if exist_in_list_image_paths:
            # print(f"image_path: {image_path}")
            valid_image_paths.append(
                (
                    image_path,
                    get_middle_frame_path(image_path, step, max_frames),
                )
            )

    return valid_image_paths


def check_caption_path(valid_image_paths, captions_dir):
    """
    """
    valid_caption_paths = []

    for image_path, middle_frame_path in valid_image_paths:
        # construct the caption path
        caption_path = os.path.join(
            captions_dir,
            os.path.basename(middle_frame_path),
        )

        # check if the caption path exists
        if os.path.exists(caption_path):
            valid_caption_paths.append(
                (
                    image_path,
                    caption_path,
                )
            )

    return valid_caption_paths


class HGVideoFeatureDataset(Dataset):
    def __init__(
        self,
        captions_y_dir,
        images_feature_dir,
        hg_images_feature_dir,
        zero_y_path,
        use_small_batch,
        sample_fps,
        max_frames,
        start_frame=None,
        type="train",
        **kwargs,
    ):
        self.use_small_batch = use_small_batch
        self.max_frames = max_frames
        self.image_feature_dir = images_feature_dir
        self.start_frame = start_frame

        # get the list of images_y files
        list_images_feature_path = glob(
            os.path.join(hg_images_feature_dir, "*.pth")
        )
        list_images_feature_path.sort()

        # hard code the original video fps
        original_fps = 24
        # compute the frame step
        frame_step = original_fps // sample_fps
        # get the valid image paths
        valid_images_feature_paths = get_valid_image_paths(
            list_images_feature_path,
            frame_step,
            max_frames
        )

        # clean the caption_y files
        self.valid_paths = check_caption_path(
            valid_images_feature_paths,
            captions_y_dir,
        )
        if self.use_small_batch:
            assert start_frame is not None
            self.valid_paths = self.valid_paths[
                start_frame: start_frame + 64
            ]

        if type == "train":
            # for application.change_background
            if start_frame is not None:
                actual_start_frame = [
                    i
                    for i, (path, _) in enumerate(self.valid_paths)
                    if f"{start_frame:06d}" in path
                ]
                assert len(actual_start_frame) == 1
                self.valid_paths = self.valid_paths[actual_start_frame[0]:]
            # for training
            else:
                self.valid_paths = self.valid_paths[5500:]
        else:
            self.valid_paths = self.valid_paths[5500:]
            actual_start_frame = [
                i
                for i, (path, _) in enumerate(self.valid_paths)
                if str(start_frame) in path
            ]
            assert len(actual_start_frame) == 1
            self.valid_paths = self.valid_paths[actual_start_frame[0]:]

        # pre-load the zero_y
        # zero_y : [1, 77, 1024]
        self.zero_y = torch.load(zero_y_path, map_location='cpu')

        self.frame_step = frame_step

    def __len__(self):
        return len(self.valid_paths)

    def _get_video_feature_data(self, start_frame_feature_path):
        """
        get the video feature data given the start_frame_feature_path
        """
        # Initialize variables
        frame_list = []

        for i in range(self.max_frames):
            image_feature_path = get_next_image_path(
                start_frame_feature_path,
                step=self.frame_step,
                num_step=i,
            )

            frame = torch.load(image_feature_path, map_location='cpu')

            # Add the frame to the list
            frame_list.append(frame)

        # video_feature_data : [1, 16, 4, 32, 32]
        video_feature_data = torch.concat(frame_list, dim=1)

        return video_feature_data

    def get_image_path(self, gh_image_path):
        """
        replace the gh_image_path with image_path
        """
        # get the base folder of the gh_image_path
        base_folder, file_path = os.path.split(gh_image_path)

        # change the path with the images_dir
        image_path = os.path.join(
            self.image_feature_dir,
            file_path,
        )

        return image_path

    def __getitem__(self, index):
        """
        get the item given the index
        """
        start_frame_feature_path, caption_y_path = self.valid_paths[index]

        # get the start_frame of original video
        start_image_feature_path = self.get_image_path(
            start_frame_feature_path
        )

        video_key = os.path.splitext(
            os.path.split(start_frame_feature_path)[1]
        )[0].split('_')[1]

        gh_video_feature_data = self._get_video_feature_data(
            start_frame_feature_path
        )

        video_feature_data = self._get_video_feature_data(
            start_image_feature_path
        )

        # caption_y : [1, 77, 1024]
        caption_y = torch.load(caption_y_path, map_location='cpu')

        return (
            gh_video_feature_data.squeeze(0),
            video_feature_data.squeeze(0),
            caption_y.squeeze(0),
            video_key,
            self.zero_y.squeeze(0),
        )
