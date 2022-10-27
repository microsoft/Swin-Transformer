# Swin Transformer

[Link to original Swin Transformer project](https://github.com/microsoft/Swin-Transformer)

## Installation Instructions

1. Set up python packages

```sh
python -m venv venv
# Activate your virtual environment somehow
source venv/bin/activate.fish 
```

CUDA 11.6

```sh
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

CUDA 11.3

```sh
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

Python packages

```sh
pip install matplotlib yacs timm einops black isort flake8 flake8-bugbear termcolor tensorboard preface opencv-python
```

2. Install Apex

```sh
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

```sh
cd kernels/window_process
python setup.py install
```

3. Download Data

We use the iNat21 dataseta available on [GitHub](https://github.com/visipedia/inat_comp/tree/master/2021)

```
mkdir -p data/inat21
cd data/inat21
mkdir compressed raw
cd compressed
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz

# pv is just a progress bar
pv val.tar.gz | tar -xz
mv val ../raw/  # if I knew how tar worked I could have it extract to raw/

pv train.tar.gz | tar -xz
mv train ../raw/
```

4. Preprocess iNat 21

Use your root data folder and your size of choice.

```
python -m data.inat preprocess /mnt/10tb/data/inat21/ val resize 224
python -m data.inat preprocess /mnt/10tb/data/inat21/ train resize 224
```
