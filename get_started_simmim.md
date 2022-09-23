## Getting Started for SimMIM

### Installation

- Install `CUDA 11.3` with `cuDNN 8` following the official installation guide of [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive).

- Setup conda environment:
```bash
# Create environment
conda create -n SimMIM python=3.8 -y
conda activate SimMIM

# Install requirements
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y

# Install apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

# Clone SimMIM
git clone https://github.com/microsoft/SimMIM
cd SimMIM

# Install other requirements
pip install -r requirements.txt
```

### Evaluating provided models

To evaluate a provided model on ImageNet validation set, run:
```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> main_simmim_ft.py \
--eval --cfg <config-file> --resume <checkpoint> --data-path <imagenet-path>
```

For example, to evaluate the `Swin Base` model on a single GPU, run:
```bash
python -m torch.distributed.launch --nproc_per_node 1 main_simmim_ft.py \
--eval --cfg configs/swin_base__800ep/simmim_finetune__swin_base__img224_window7__800ep.yaml --resume simmim_finetune__swin_base__img224_window7__800ep.pth --data-path <imagenet-path>
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
--cfg configs/swin_base__800ep/simmim_pretrain__swin_base__img192_window6__800ep.yaml --batch-size 128 --data-path <imagenet-path>/train [--output <output-directory> --tag <job-tag>]
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
--cfg configs/swin_base__800ep/simmim_finetune__swin_base__img224_window7__800ep.yaml --batch-size 128 --data-path <imagenet-path> --pretrained <pretrained-ckpt> [--output <output-directory> --tag <job-tag>]
```