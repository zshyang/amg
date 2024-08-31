<p align="center">

  <h2 align="center">AMG: Avatar Motion Guided Video Generation</h2>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=VaRp0cMAAAAJ&hl=en"><strong>Zhangsihao Yang</strong></a><sup>1</sup>
    ·  
    <a href="https://shanmy.github.io/"><strong>Mengyi Shan</strong></a><sup>2</sup>
    ·
    <a href=""><strong>Mohammad Farazi</strong></a><sup>1</sup>
    ·
    <a href=""><strong>Wenhui Zhu</strong></a><sup>1</sup>
    ·
    <a href=""><strong>Yanxi Chen</strong></a><sup>1</sup>
    ·
    <a href=""><strong>Xuanzhao Dong</strong></a><sup>1</sup>
    ·
    <a href=""><strong>Yalin Wang</strong></a><sup>1</sup>
    <br>
    <sup>1</sup>Arizona State University &nbsp;&nbsp;&nbsp; <sup>2</sup>University of Washington
    </br>
        <!-- <a href="https://arxiv.org/abs/2403.09069">
        <img src='https://img.shields.io/badge/arXiv-DIM-green' alt='Paper PDF'>
        </a> -->
        <a href='https://zshyang.github.io/amg-website/'>
        <img src='https://img.shields.io/badge/Project_Page-AMG-blue' alt='Project Page'></a>
        <!-- <a href='https://youtu.be/VPJe6TyrT-Y'>
        <img src='https://img.shields.io/badge/YouTube-MagicPose-rgb(255, 0, 0)' alt='Youtube'></a> -->
     </br>
    <!-- <table align="center">
        <img src="./assets/demo1.gif">
        <img src="./assets/demo2.gif">
    </table> -->
</p>

_Human video generation is a challenging task due to the complexity of human body movements and the need for photorealism. While 2D methods excel in realism, they lack 3D control, and 3D avatar-based approaches struggle with seamless background integration. We introduce AMG, a method that merges 2D photorealism with 3D control by conditioning video diffusion models on 3D avatar renderings. AMG enables multi-person video generation with precise control over camera positions, human motions, and background style, outperforming existing methods in realism and adaptability._

## Getting Started

Make `vgen` virual environment.

Download [model.zip](https://drive.google.com/file/d/1n979-fIwIBlxqavI_lJQFFrMUKcJwqjI/view?usp=sharing) to `_runtime` and unzip it.

## Inference

The weights for inference could be downloaded from [here](https://drive.google.com/file/d/1g274tXyfaA45cy8IkaUJF39iVg5sQNTU/view?usp=sharing) (5.28GB).

Run the following command line for installing `amg` package first.

```bash
pip install -e .
```

### Change Background

Run the command below to get **change background** results:

```bash
python applications/change_background.py --cfg configs/applications/change_background/demo.yaml
```

The results are store under newly created folder `_demo_results`.
You should be able to see exact same results like the following:

<table align="center">
    <tr>
        <th style="text-align:center;">Input</th>
        <th style="text-align:center;">Reference</th>
        <th style="text-align:center;">Generated</th>
    </tr>
    <tr>
        <td colspan="3" align="center">
            <img src="./doc/change_background.gif" alt="GIF description">
        </td>
    </tr>
</table>

### Move Camera

Run the command below to get **move camera** results:

```bash
python applications/move_camera.py --cfg configs/applications/move_camera/demo.yaml
```

## Traning

### Download data

1. Fill out [this google form](https://forms.gle/xrx4sfAn7QAWgiXq9) for reqeusting of the processed dataset in the paper.

2. Put the downloaded data under `_data`.

## Folder Structure

- configs
- demo_data
