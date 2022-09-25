# Swin Transformer for Image Classification

This folder contains the implementation of the Swin Transformer for image classification.

## Model Zoo

Please refer to [MODEL HUB](MODELHUB.md#imagenet-22k-pretrained-swin-moe-models) for more pre-trained models.

## Usage

### Install

We recommend using the pytorch docker `nvcr>=21.05` by
nvidia: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch.

- Clone this repo:

```bash
git clone https://github.com/microsoft/Swin-Transformer.git
cd Swin-Transformer
```

- Create a conda virtual environment and activate it:

```bash
conda create -n swin python=3.7 -y
conda activate swin
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.8.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
```

- Install `timm==0.4.12`:

```bash
pip install timm==0.4.12
```

- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
```

- Install fused window process for acceleration, activated by passing `--fused_window_process` in the running script
```bash
cd kernels/window_process
python setup.py install #--user
```

### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train_map.txt`, `val_map.txt`: which store the relative path in the corresponding zip file and ground truth
      label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip
  
  $ head -n 5 data/ImageNet-Zip/val_map.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516
  
  $ head -n 5 data/ImageNet-Zip/train_map.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```
- For ImageNet-22K dataset, make a folder named `fall11_whole` and move all images to labeled sub-folders in this
  folder. Then download the train-val split
  file ([ILSVRC2011fall_whole_map_train.txt](https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_train.txt)
  & [ILSVRC2011fall_whole_map_val.txt](https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_val.txt))
  , and put them in the parent directory of `fall11_whole`. The file structure should look like:

  ```bash
    $ tree imagenet22k/
    imagenet22k/
    ├── ILSVRC2011fall_whole_map_train.txt
    ├── ILSVRC2011fall_whole_map_val.txt
    └── fall11_whole
        ├── n00004475
        ├── n00005787
        ├── n00006024
        ├── n00006484
        └── ...
  ```

### Evaluation

To evaluate a pre-trained `Swin Transformer` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --resume <checkpoint> --data-path <imagenet-path> 
```

For example, to evaluate the `Swin-B` with a single GPU:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
--cfg configs/swin/swin_base_patch4_window7_224.yaml --resume swin_base_patch4_window7_224.pth --data-path <imagenet-path>
```

### Training from scratch on ImageNet-1K

To train a `Swin Transformer` on ImageNet from scratch, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \ 
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

**Notes**:

- To use zipped ImageNet instead of folder dataset, add `--zip` to the parameters.
    - To cache the dataset in the memory instead of reading from files every time, add `--cache-mode part`, which will
      shard the dataset into non-overlapping pieces for different GPUs and only load the corresponding one for each GPU.
- When GPU memory is not enough, you can try the following suggestions:
    - Use gradient accumulation by adding `--accumulation-steps <steps>`, set appropriate `<steps>` according to your need.
    - Use gradient checkpointing by adding `--use-checkpoint`, e.g., it saves about 60% memory when training `Swin-B`.
      Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.
    - We recommend using multi-node with more GPUs for training very large models, a tutorial can be found
      in [this page](https://pytorch.org/tutorials/intermediate/dist_tuto.html).
- To change config options in general, you can use `--opts KEY1 VALUE1 KEY2 VALUE2`, e.g.,
  `--opts TRAIN.EPOCHS 100 TRAIN.WARMUP_EPOCHS 5` will change total epochs to 100 and warm-up epochs to 5.
- For additional options, see [config](config.py) and run `python main.py --help` to get detailed message.

For example, to train `Swin Transformer` with 8 GPU on a single node for 300 epochs, run:

`Swin-T`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 128 
```

`Swin-S`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/swin/swin_small_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 128 
```

`Swin-B`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/swin/swin_base_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 64 \
--accumulation-steps 2 [--use-checkpoint]
```

### Pre-training on ImageNet-22K

For example, to pre-train a `Swin-B` model on ImageNet-22K:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/swin/swin_base_patch4_window7_224_22k.yaml --data-path <imagenet22k-path> --batch-size 64 \
--accumulation-steps 8 [--use-checkpoint]
```

### Fine-tuning on higher resolution

For example, to fine-tune a `Swin-B` model pre-trained on 224x224 resolution to 384x384 resolution:

```bashs
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/swin/swin_base_patch4_window12_384_finetune.yaml --pretrained swin_base_patch4_window7_224.pth \
--data-path <imagenet-path> --batch-size 64 --accumulation-steps 2 [--use-checkpoint]
```

### Fine-tuning from a ImageNet-22K(21K) pre-trained model

For example, to fine-tune a `Swin-B` model pre-trained on ImageNet-22K(21K):

```bashs
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/swin/swin_base_patch4_window7_224_22kto1k_finetune.yaml --pretrained swin_base_patch4_window7_224_22k.pth \
--data-path <imagenet-path> --batch-size 64 --accumulation-steps 2 [--use-checkpoint]
```

### Throughput

To measure the throughput, run:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg <config-file> --data-path <imagenet-path> --batch-size 64 --throughput --disable_amp
```


## Mixture-of-Experts Support

### Install [Tutel](https://github.com/microsoft/tutel)
```bash
python3 -m pip uninstall tutel -y 
python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@main
```

### Training Swin-MoE 
For example, to train a `Swin-MoE-S` model with 32 experts on ImageNet-22K with 32 GPUs (4 nodes):

```bash
python -m torch.distributed.launch --nproc_per_node 8 --nnode=4 \
--node_rank=<node-rank> --master_addr=<master-ip> --master_port 12345  main_moe.py \
--cfg configs/swinmoe/swin_moe_small_patch4_window12_192_32expert_32gpu_22k.yaml --data-path <imagenet22k-path> --batch-size 128
```

### Evaluating Swin-MoE

To evaluate a `Swin-MoE-S` with 32 experts on ImageNet-22K with 32 GPUs (4 nodes):

1. Download the zip file [swin_moe_small_patch4_window12_192_32expert_32gpu_22k.zip](https://github.com/SwinTransformer/storage/releases/download/v2.0.2/swin_moe_small_patch4_window12_192_32expert_32gpu_22k.zip) which contains the pre-trained models for each rank, and unzip them to the folder "swin_moe_small_patch4_window12_192_32expert_32gpu_22k".
2. Run the following evaluation command, note the checkpoint path should not contain the ".rank\<x\>" suffix.

```bash
python -m torch.distributed.launch --nproc_per_node 8 --nnode=4 \
--node_rank=<node-rank> --master_addr=<master-ip> --master_port 12345  main_moe.py \
--cfg configs/swinmoe/swin_moe_small_patch4_window12_192_32expert_32gpu_22k.yaml --data-path <imagenet22k-path> --batch-size 128 \
--resume swin_moe_small_patch4_window12_192_32expert_32gpu_22k/swin_moe_small_patch4_window12_192_32expert_32gpu_22k.pth 
```

More Swin-MoE models can be found in [MODEL HUB](MODELHUB.md#imagenet-22k-pretrained-swin-moe-models)

## SimMIM Support

### Evaluating provided models

To evaluate a provided model on ImageNet validation set, run:
```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> main_simmim_ft.py \
--eval --cfg <config-file> --resume <checkpoint> --data-path <imagenet-path>
```

For example, to evaluate the `Swin Base` model on a single GPU, run:
```bash
python -m torch.distributed.launch --nproc_per_node 1 main_simmim_ft.py \
--eval --cfg configs/simmim/simmim_finetune__swin_base__img224_window7__800ep.yaml --resume simmim_finetune__swin_base__img224_window7__800ep.pth --data-path <imagenet-path>
```

### Pre-training with SimMIM
To pre-train models with `SimMIM`, run:
```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> main_simmim_pt.py \ 
--cfg <config-file> --data-path <imagenet-path>/train [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

For example, to pre-train `Swin Base` for 800 epochs on one DGX-2 server, run:
```bash
python -m torch.distributed.launch --nproc_per_node 16 main_simmim_pt.py \ 
--cfg configs/simmim/simmim_pretrain__swin_base__img192_window6__800ep.yaml --batch-size 128 --data-path <imagenet-path>/train [--output <output-directory> --tag <job-tag>]
```

### Fine-tuning pre-trained models
To fine-tune models pre-trained by `SimMIM`, run:
```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> main_simmim_ft.py \ 
--cfg <config-file> --data-path <imagenet-path> --pretrained <pretrained-ckpt> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

For example, to fine-tune `Swin Base` pre-trained by `SimMIM` on one DGX-2 server, run:
```bash
python -m torch.distributed.launch --nproc_per_node 16 main_simmim_ft.py \ 
--cfg configs/simmim/simmim_finetune__swin_base__img224_window7__800ep.yaml --batch-size 128 --data-path <imagenet-path> --pretrained <pretrained-ckpt> [--output <output-directory> --tag <job-tag>]
```