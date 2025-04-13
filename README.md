# CMLab_Gembench_SAM2Act
gembench with sam2act

# Installation Instructions

1. Install general python packages
```bash
conda create -n gembench python==3.10

conda activate gembench

# On CLEPS, first run `module load gnu12/12.2.0`

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
# FORCE_CUDA=1 pip install torch-scatter==2.1.2
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu118.html

### if you using 11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
### if you using 11.8

export CUDA_HOME=$HOME/anaconda3/envs/gembench
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

cd robot-3dlotus
pip install -r requirements.txt

# install genrobo3d
pip install -e .
```

2. Install RLBench
```bash
mkdir dependencies
cd dependencies
```

Download CoppeliaSim (see instructions [here](https://github.com/stepjam/PyRep?tab=readme-ov-file#install))
```bash
# change the version if necessary
wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```

Add the following to your ~/.bashrc file:
```bash
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0
```

Install Pyrep and RLBench
```bash
git clone https://github.com/cshizhe/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install .
cd ..

# Our modified version of RLBench to support new tasks in GemBench
git clone https://github.com/rjgpinel/RLBench
cd RLBench
pip install -r requirements.txt
pip install .
cd ../..
```

3. Install model dependencies

```bash
cd dependencies

# Please ensure to set CUDA_HOME beforehand as specified in the export const of the section 1
git clone https://github.com/cshizhe/chamferdist.git
cd chamferdist
python setup.py install
cd ..

git clone https://github.com/cshizhe/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib
python setup.py install
cd ../..

# llama3: needed for 3D-LOTUS++
git clone https://github.com/cshizhe/llama3.git
cd llama3
pip install -e .
cd ../..
```

### Install SAM2ACT ###
```
cd SAM2Act
```
Install PyTorch3D.

The original RVT repository recommends that you can skip this step if you just want to use RVT-2 backbone and its custom Point-Renderer for rendering. If you want to try out RVT backbone or different renderer, PyTorch3D is required. However, we still recommend installing this as there are some nested dependencies that requires PyTorch3D package.

One recommended version that is compatible with the rest of the library can be installed as follows. Note that this might take some time. For more instructions visit [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
```
FORCE_CUDA=1 pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
```

install sam2act and other submodules.

To locally install the repository, you can either `pip install -e '.[xformers]'` to install the library with [xformers](https://github.com/facebookresearch/xformers) or `pip install -e .` to install without it. We recommend using the former as improves speed. However, sometimes the installation might fail due to the xformers dependency. In that case, you can install the library without xformers. The performance difference between the two is minimal but speed could be slower without xformers.
```
pip install -e '.[xformers]' 
```
Note that for bug-free implementation, we still suggest installing without `xformers` as below.
```
pip install -e .
```
Install, required libraries for  YARR, PerAct Colab, and Point Renderer.
```
pip install -e sam2act/libs/YARR 
pip install -e sam2act/libs/PERACT_COLAB
pip install -e sam2act/libs/point-renderer
``` 

add SAM2ACT PythonPath
```
export PYTHONPATH=/home/bill/Documents/research/CVPR_gembench_baseline/robot-3dlotus/SAM2Act/sam2act:$PYTHONPATH
```

You may also want to upgrade some packages if there is any error:
```
pip install --upgrade hydra-core
``` 
 
4. Download SAM2 weights and dataset.
    - Before starting, download SAM2 pretrained weights for loading SAM2Act using the following command.
    ```
    cd sam2act/mvt/sam2_train/checkpoints
    download_ckpts.sh
    ``` 

    - For experiments on RLBench, we use [pre-generated dataset](https://drive.google.com/drive/folders/0B2LlLwoO3nfZfkFqMEhXWkxBdjJNNndGYl9uUDQwS1pfNkNHSzFDNGwzd1NnTmlpZXR1bVE?resourcekey=0-jRw5RaXEYRLe2W6aNrNFEQ) provided by [PerAct](https://github.com/peract/peract#download). If downloading from Google Drive encounters any limits, we also provide a mirror of the same dataset in [Hugging Face](https://huggingface.co/datasets/hqfang/RLBench-18-Tasks). Please download and place them under `SAM2Act/sam2act/data/xxx` where `xxx` is either `train`, `test`, or `val`.  

    - Additionally, building upon PerAct's dataloader, we create a new dataloader that can sample observation sequences of a given size, and it also supports the same functionality as PerAct's. Same as PerAct, our dataloader is also based on [YARR](https://github.com/stepjam/YARR). YARR creates a replay buffer on the fly which can increase the startup time. We provide an option to directly load the replay buffer from the disk. We recommend using the pre-generated replay buffer (98 GB) as it reduces the startup time. You can download replay buffer for [indidual tasks](https://huggingface.co/datasets/hqfang/SAM2Act/tree/main/replay_temporal/replay_train). After downloading, uncompress the replay buffer(s) (for example using the command `tar -xf <task_name>.tar.xz`) and place it under `SAM2Act/sam2act/replay_temporal/replay_xxx` where `xxx` is `train` (for now we only provide replay buffer for trianing split). Note that is useful only if you want to train SAM2Act from scratch and not needed if you want to evaluate the pre-trained model.

    - If you prefer using dataloader same as PerAct's, you can refer to the step 6 of this [instruction](https://github.com/NVlabs/RVT?tab=readme-ov-file#getting-started). You also need to change `get_dataset_temporal` to `get_dataset` in `train.py`. Again, note that this is not necessary because our dataloader preserves all functionality of PerAct's.

    - For experiments on MemoryBench, we also provide a [pre-generated dataset](https://huggingface.co/datasets/hqfang/MemoryBench). Please download and place them under `SAM2Act/sam2act/data_memory/xxx` where `xxx` is either `train` or `test`.  



    ### FAQ ###
    OSError: /home/bill/anaconda3/envs/CVPR_baseline_gembench/lib/python3.10/site-packages/torch_scatter/_version_cuda.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb

    ```
    FORCE_CUDA=1 pip install torch_scatter==2.1.2
    ```


    assert flash_attn is not None, "Make sure flash_attn is installed."
    AssertionError: Make sure flash_attn is installed.

    ```
    pip install flash_attn==2.5.9.post1
    ```



while using ```pip install -e sam2act/libs/point-renderer```

    The detected CUDA version (11.8) mismatches the version that was used to compile
    PyTorch (12.1). Please make sure to use the same CUDA versions.


    ```
    pip uninstall nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nvtx-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12

    ```
    from torch._C import *  # noqa: F403
ImportError: libnccl.so.2: cannot open shared object file: No such file or directory

```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" | \
  sudo tee /etc/apt/sources.list.d/cuda.list

  sudo apt update
sudo apt install libnccl2 libnccl-dev
```


assert flash_attn is not None, "Make sure flash_attn is installed."
AssertionError: Make sure flash_attn is installed.
```
pip install ninja cmake
pip uninstall -y flash-attn
pip cache purge
python -m pip install --upgrade pip wheel setuptools
MAX_JOBS=8 python -m pip -v install flash-attn --no-build-isolation

if occur some error like Failed to build flash-attn
ERROR: Failed to build installable wheels for some pyproject.toml
## reduce MAX_JOBS num ##
```
reference :https://github.com/Dao-AILab/flash-attention/issues/1038




```
pip uninstall opencv-python
pip install opencv-python-headless
```